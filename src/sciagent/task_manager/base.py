from typing import Any, Dict, Optional, Callable, Literal, List, Sequence
import json
import sqlite3
import logging
import time
from pathlib import Path

from sciagent.message_proc import print_message
from sciagent.api.memory import MemoryManagerConfig
from sciagent.message_proc import (
    generate_openai_message, 
    get_message_elements_as_text, 
    has_tool_call,
    get_tool_call_info,
    purge_context_images,
    complete_unresponded_tool_calls
)
from sciagent.tool.base import BaseTool
from sciagent.tool.mcp import MCPTool
from sciagent.agent.openai import OpenAIAgent
from sciagent.agent.argo import ArgoAgent
from sciagent.agent.base import BaseAgent
from sciagent.util import get_timestamp
from sciagent.tool.base import ToolReturnType
from sciagent.tool.coding import PythonCodingTool, BashCodingTool
from sciagent.skill import SkillTool, load_skills, extract_markdown_image_paths
from sciagent.api.llm_config import LLMConfig, OpenAIConfig, AskSageConfig, ArgoConfig
from sciagent.agent.memory import MemoryQueryResult, VectorStore
from sciagent.exceptions import MaxRoundsReached
try:
    from sciagent.agent.asksage import AskSageAgent
except ImportError:
    logging.warning(
        "AskSage endpoint is not supported because the `asksageclient` "
        "package is not installed. To use AskSage endpoints, install it with "
        "`pip install asksageclient`."
    )
    AskSageAgent = None

logger = logging.getLogger(__name__)


class AgentGroup(list[BaseAgent]):

    @property
    def default(self) -> Optional[BaseAgent]:
        return self[0] if self else None

    def get_by_name(self, name: str) -> Optional[BaseAgent]:
        for agent in self:
            if agent.name == name:
                return agent
        return None


class BaseTaskManager:

    assistant_system_message = ""
            
    def __init__(
        self, 
        llm_config: LLMConfig = None,
        memory_config: Optional[MemoryManagerConfig] = None,
        tools: list[BaseTool] = (), 
        skill_dirs: Optional[Sequence[str]] = None,
        message_db_path: Optional[str] = None,
        fill_context_with_message_db: bool = False,
        use_webui: bool = False,
        use_coding_tools: bool = True,
        run_codes_in_sandbox: bool = False,
        build: bool = True,
        *args,
        memory_vector_store: Optional[VectorStore] = None,
        memory_notability_filter: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
        memory_formatter: Optional[Callable[[List[MemoryQueryResult]], str]] = None,
        memory_embedder: Optional[Callable[[Sequence[str]], List[List[float]]]] = None,
        **kwargs
    ):
        """The base task manager.
        
        Parameters
        ----------
        llm_config : LLMConfig
            The configuration for the LLM.
        memory_config : MemoryManagerConfig, optional
            Optional configuration for long-term memory. Use
            `MemoryManagerConfig.from_dict` to build from a mapping if needed.
        memory_vector_store, memory_notability_filter, memory_formatter, memory_embedder : optional
            Overrides propagated to the agent for custom memory behaviour.
        tools : list[BaseTool]
            A list of tools provided to the agent.
        skill_dirs : Optional[Sequence[str]]
            Directories that contain skill folders with SKILL.md definitions.
        message_db_path : Optional[str]
            If provided, the entire chat history will be stored in 
            a SQLite database at the given path.
        fill_context_with_message_db : bool
            If True and `message_db_path` is provided, load the message database
            and populate the task manager context after the system message.
        use_webui : bool
            If True, user input will be received from the WebUI via
            the message database. This requires `message_db_path`.
        use_coding_tools : bool
            If True, register default Python and Bash coding tools.
        run_codes_in_sandbox : bool
            If True, execute default coding tools in a container sandbox.
        build : bool
            Whether to build the internal state of the task manager.
        """
        self.context = []
        self.full_history = []
        self.llm_config = llm_config
        if isinstance(memory_config, dict):  # type: ignore[unreachable]
            memory_config = MemoryManagerConfig.from_dict(memory_config)  # pragma: no cover - legacy path
        self.memory_config = memory_config
        self.agent_group = AgentGroup()
        self.tools = tools
        self.skill_dirs = list(skill_dirs) if skill_dirs else []
        self.skill_catalog = []
        
        if use_webui and not message_db_path:
            raise ValueError("`use_webui` requires `message_db_path` to be set.")

        self.use_webui = use_webui
        self.use_coding_tools = use_coding_tools
        self.run_codes_in_sandbox = run_codes_in_sandbox
        self.message_db_path = message_db_path
        self.fill_context_with_message_db = fill_context_with_message_db
        self.message_db_conn = None
        self.webui_user_input_last_timestamp = 0

        self._memory_vector_store = memory_vector_store
        self._memory_notability_filter = memory_notability_filter
        self._memory_formatter = memory_formatter
        self._memory_embedder = memory_embedder
        
        if build:
            self.build()

    @property
    def agent(self) -> Optional[BaseAgent]:
        return self.agent_group.default

    @agent.setter
    def agent(self, agent: Optional[BaseAgent]) -> None:
        if agent is None:
            self.agent_group = AgentGroup()
            return
        if self.agent_group:
            self.agent_group[0] = agent
        else:
            self.agent_group.append(agent)
        
    def build(self, *args, **kwargs):
        self.build_db()
        self.build_agent()
        self.build_tools()
    
    def build_db(self, *args, **kwargs):
        if self.message_db_path:
            self.message_db_conn = sqlite3.connect(self.message_db_path)
            self.message_db_conn.execute(
                "CREATE TABLE IF NOT EXISTS messages (timestamp TEXT, role TEXT, content TEXT, tool_calls TEXT, image TEXT)"
            )
            self.message_db_conn.execute(
                "CREATE TABLE IF NOT EXISTS status (id INTEGER PRIMARY KEY, user_input_requested INTEGER)"
            )
            cursor = self.message_db_conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM status")
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    "INSERT INTO status (id, user_input_requested) VALUES (1, 0)"
                )
            self.message_db_conn.commit()
            
            # Set timestamp buffer to the timestamp of the last user input in the database
            # if it exists..
            if self.use_webui:
                cursor.execute(
                    "SELECT timestamp, role, content, tool_calls, image FROM messages "
                    "WHERE role = 'user_webui' ORDER BY rowid"
                )
                messages = cursor.fetchall()
                if len(messages) > 0 and self.webui_user_input_last_timestamp == 0:
                    self.webui_user_input_last_timestamp = int(messages[-1][0])
            if self.fill_context_with_message_db:
                self._fill_context_from_message_db()

    def _fill_context_from_message_db(self) -> None:
        if not self.message_db_conn:
            return
        cursor = self.message_db_conn.cursor()
        cursor.execute(
            "SELECT rowid, timestamp, role, content, tool_calls, image FROM messages ORDER BY rowid"
        )
        rows = cursor.fetchall()
        if not rows:
            return

        pending_tool_call_ids: list[str] = []
        messages: list[dict[str, Any]] = []
        for _, _, role, content, tool_calls, image in rows:
            normalized_role = "user" if role == "user_webui" else role
            content_text = (content or "").strip()
            if content_text:
                content_text = "\n".join(
                    line for line in content_text.splitlines() if line.strip() != "<image>"
                ).strip()

            message: dict[str, Any] = {"role": normalized_role, "content": content_text}

            if image:
                image_url = image
                if isinstance(image_url, bytes):
                    image_url = image_url.decode("utf-8", errors="ignore")
                if not image_url.startswith("data:image"):
                    image_url = f"data:image/png;base64,{image_url}"
                message["content"] = [
                    {"type": "text", "text": content_text},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]

            tool_call_list = self._parse_tool_calls_text(tool_calls)
            if tool_call_list:
                message["tool_calls"] = tool_call_list
                pending_tool_call_ids.extend(
                    [tool_call["id"] for tool_call in tool_call_list if "id" in tool_call]
                )
            elif normalized_role == "tool" and pending_tool_call_ids:
                message["tool_call_id"] = pending_tool_call_ids.pop(0)

            messages.append(message)

        insert_index = 0
        while (
            insert_index < len(self.context)
            and self.context[insert_index].get("role") == "system"
        ):
            insert_index += 1
        for offset, message in enumerate(messages):
            self.context.insert(insert_index + offset, message)

        history_insert_index = 0
        while (
            history_insert_index < len(self.full_history)
            and self.full_history[history_insert_index].get("role") == "system"
        ):
            history_insert_index += 1
        for offset, message in enumerate(messages):
            self.full_history.insert(history_insert_index + offset, message)

    def _parse_tool_calls_text(self, tool_calls: Optional[str]) -> list[dict[str, Any]]:
        if not tool_calls:
            return []
        lines = [line.strip() for line in tool_calls.splitlines() if line.strip()]
        calls: list[dict[str, Any]] = []
        i = 0
        while i < len(lines):
            header = lines[i]
            if ":" not in header:
                i += 1
                continue
            tool_id, tool_name = header.split(":", 1)
            tool_id = tool_id.strip()
            tool_name = tool_name.strip()
            arguments = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("Arguments:"):
                arguments = lines[i + 1].split("Arguments:", 1)[1].strip()
                i += 2
            else:
                i += 1
            calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": arguments},
                }
            )
        return calls
    
    def build_agent(self, *args, **kwargs):
        """Build the assistant(s)."""
        if self.llm_config is None:
            logger.info(
                "Skipping agent build because `llm_config` is not provided."
            )
            return
        agent = self._create_agent(self.llm_config)
        if agent is not None:
            self.agent_group = AgentGroup([agent])

    def add_agent(
        self,
        llm_config: Optional[LLMConfig] = None,
        name: Optional[str] = None,
    ) -> Optional[BaseAgent]:
        if llm_config is None:
            default_agent = self.agent
            if default_agent is not None:
                llm_config = default_agent.llm_config
            else:
                llm_config = self.llm_config
        if llm_config is None:
            logger.info(
                "Skipping agent build because `llm_config` is not provided."
            )
            return None
        agent_name = name or "assistant"
        if self.agent_group.get_by_name(agent_name) is not None:
            raise ValueError(f"An agent named '{agent_name}' already exists.")
        agent = self._create_agent(llm_config, name=agent_name)
        if agent is None:
            return None
        self.agent_group.append(agent)
        tools = self._collect_base_tools()
        if tools:
            self.register_tools(tools, agent=agent)
        return agent

    def _create_agent(self, llm_config: LLMConfig, name: str = "assistant") -> BaseAgent:
        if not isinstance(llm_config, LLMConfig):
            raise ValueError(
                "`llm_config` must be an instance of `LLMConfig`. The type of this "
                "config object will be used to determine the API type of the LLM."
            )
        agent_class = {
            OpenAIConfig: OpenAIAgent,
            AskSageConfig: AskSageAgent,
            ArgoConfig: ArgoAgent,
        }[type(llm_config)]
        if agent_class is None:
            raise RuntimeError(
                f"Dependencies required for {agent_class.__name__} is unavailable."
            )
        agent = agent_class(
            llm_config=llm_config,
            name=name,
            system_message=self.assistant_system_message,
            memory_config=self.memory_config,
            memory_vector_store=self._memory_vector_store,
            memory_notability_filter=self._memory_notability_filter,
            memory_formatter=self._memory_formatter,
            memory_embedder=self._memory_embedder,
        )
        agent.set_tool_approval_handler(self._request_tool_approval_via_task_manager)
        return agent
    
    def build_tools(self, *args, **kwargs):
        if self.agent is not None:
            tools = self._collect_base_tools()
            self.register_tools(tools)

    def _collect_base_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = list(self.tools)
        self._merge_tools(tools, self._build_default_tools())
        self._merge_tools(tools, self._build_skill_tools())
        return tools

    def _merge_tools(self, tools: list[BaseTool], new_tools: list[BaseTool]) -> None:
        seen_names = self._collect_tool_names(tools)
        for tool in new_tools:
            tool_names = self._collect_tool_names([tool])
            if tool_names and tool_names & seen_names:
                continue
            tools.append(tool)
            seen_names.update(tool_names)

    def _collect_tool_names(self, tools: list[BaseTool]) -> set[str]:
        names: set[str] = set()
        for tool in tools:
            if isinstance(tool, MCPTool):
                continue
            for exposed in tool.exposed_tools:
                names.add(exposed.name)
        return names

    def _build_default_tools(self) -> list[BaseTool]:
        if not self.use_coding_tools:
            return []
        return [
            PythonCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
            BashCodingTool(run_in_sandbox=self.run_codes_in_sandbox),
        ]

    def _build_skill_tools(self) -> list[BaseTool]:
        if not self.skill_dirs:
            self.skill_catalog = []
            return []
        skills = load_skills(self.skill_dirs)
        self.skill_catalog = skills
        return [SkillTool(skill) for skill in skills]

    def _parse_tool_response_payload(self, content: Any) -> Optional[Dict[str, Any]]:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return None

        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _build_skill_doc_messages(
        self,
        tool_response: Dict[str, Any],
        tool_call_info: Optional[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        if tool_call_info is None:
            return []

        function_info = tool_call_info.get("function", {})
        tool_name = function_info.get("name")
        skill_tool_names = {skill.tool_name for skill in self.skill_catalog}
        if tool_name not in skill_tool_names:
            return []

        payload = self._parse_tool_response_payload(tool_response.get("content"))
        if payload is None:
            return []

        files = payload.get("files")
        if not isinstance(files, dict):
            return []

        skill_root = payload.get("path")
        skill_root_path = Path(skill_root) if isinstance(skill_root, str) else None
        images_by_file_raw = payload.get("images_by_file")
        images_by_file = images_by_file_raw if isinstance(images_by_file_raw, dict) else {}

        messages: list[Dict[str, Any]] = []
        for relative_path, file_content in files.items():
            if not isinstance(relative_path, str) or not isinstance(file_content, str):
                continue

            text_message = generate_openai_message(
                content=f"[Skill file: {relative_path}]\n{file_content}",
                role="user",
            )
            messages.append(text_message)

            image_paths = images_by_file.get(relative_path, [])
            parsed_image_paths: list[str] = []
            if isinstance(image_paths, list):
                parsed_image_paths = [path for path in image_paths if isinstance(path, str)]

            if not parsed_image_paths:
                markdown_path = None
                if skill_root_path is not None:
                    markdown_path = skill_root_path / relative_path
                parsed_image_paths = extract_markdown_image_paths(
                    file_content,
                    markdown_path=markdown_path,
                )

            for image_path in parsed_image_paths:
                try:
                    image_message = generate_openai_message(
                        content=f"Image referenced by skill file `{relative_path}`.",
                        role="user",
                        image_path=image_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load image '%s' referenced by skill file '%s': %s",
                        image_path,
                        relative_path,
                        exc,
                    )
                    image_message = generate_openai_message(
                        content=f"Image path from skill file `{relative_path}`: {image_path}",
                        role="user",
                    )
                messages.append(image_message)

        return messages

    def _inject_skill_doc_messages_to_context(
        self,
        tool_response: Dict[str, Any],
        tool_call_info: Optional[Dict[str, Any]],
    ) -> None:
        skill_doc_messages = self._build_skill_doc_messages(tool_response, tool_call_info)
        for skill_doc_message in skill_doc_messages:
            self.update_message_history(
                skill_doc_message,
                update_context=True,
                update_full_history=True,
            )

    def register_tools(
        self,
        tools: BaseTool | list[BaseTool],
        agent: Optional[BaseAgent] = None,
    ) -> None:
        if not isinstance(tools, (list, tuple)):
            tools = [tools]
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ValueError("Input should be a list of BaseTool objects.")
            if (
                not isinstance(tool, MCPTool) and
                (
                    not hasattr(tool, "exposed_tools")
                    or (hasattr(tool, "exposed_tools") and len(tool.exposed_tools) == 0)
                )
            ):
                raise ValueError(
                    "A subclass of BaseTool that is not MCPTool must provide at least one ExposedToolSpec in `exposed_tools`."
                )
        target_agents = [agent] if agent is not None else list(self.agent_group)
        for target in target_agents:
            target.register_tools(list(tools))

    def prerun_check(self, *args, **kwargs) -> bool:
        return True
    
    def record_system_message(
        self, content: str, image_path: Optional[str] = None, update_context: bool = False
    ):
        """Add a system message to the message history.
        """
        message = generate_openai_message(
            content=content,
            role="system",
            image_path=image_path
        )
        self.update_message_history(
            message,
            update_context=update_context,
            update_full_history=True,
        )
    
    def update_message_history(
        self,
        message: Dict[str, Any],
        update_context: bool = True,
        update_full_history: bool = True,
        update_db: bool = True
    ) -> None:
        if update_context:
            self.context.append(message)
        if update_full_history:
            self.full_history.append(message)
        if self.message_db_conn and update_db:
            self.add_message_to_db(message)
            
    def add_message_to_db(self, message: Dict[str, Any]) -> None:
        elements = get_message_elements_as_text(message)
        self.message_db_conn.execute(
            "INSERT INTO messages (timestamp, role, content, tool_calls, image) VALUES (?, ?, ?, ?, ?)",
            (
                str(get_timestamp(as_int=True)), 
                elements["role"], 
                elements["content"], 
                elements["tool_calls"], 
                elements["image"]
            )
        )
        self.message_db_conn.commit()
        
    def get_user_input(
        self, 
        prompt: str = "Enter a message: ",
        display_prompt_in_webui: bool = False,
        *args, **kwargs
    ) -> str:
        """Get user input. If WebUI mode is enabled, input is relayed by the
        SQL message database. Otherwise, the user will be prompted to enter a
        message from terminal.
        
        Parameters
        ----------
        prompt : Optional[str], optional
            The prompt to display to the user in the terminal.
        display_prompt_in_webui : bool, optional
            If True, the prompt will be displayed in the WebUI.

        Returns
        -------
        str
            The user input.
        """
        if self.use_webui:
            if not self.message_db_conn:
                raise RuntimeError("WebUI input requires an active message database connection.")
            logger.info("Getting user input from relay database. Please enter your message in the WebUI.")
            self.set_user_input_requested(True)
            cursor = self.message_db_conn.cursor()
            if display_prompt_in_webui:
                self.add_message_to_db({"role": "system", "content": prompt})
            while True:
                cursor.execute("SELECT timestamp, role, content, tool_calls, image FROM messages WHERE role = 'user_webui' ORDER BY rowid")
                messages = cursor.fetchall()
                if len(messages) > 0 and int(messages[-1][0]) > self.webui_user_input_last_timestamp:
                    self.webui_user_input_last_timestamp = int(messages[-1][0])
                    self.set_user_input_requested(False)
                    return messages[-1][2]
                time.sleep(1)
        else:
            self.set_user_input_requested(True)
            message = input(prompt)
            self.set_user_input_requested(False)
            return message
        
    def _sync_webui_user_input_last_timestamp(self) -> None:
        if self.use_webui and self.message_db_conn:
            cursor = self.message_db_conn.cursor()
            cursor.execute(
                "SELECT timestamp, role, content, tool_calls, image FROM messages WHERE role = 'user_webui' ORDER BY rowid"
            )
            messages = cursor.fetchall()
            if len(messages) > 0:
                self.webui_user_input_last_timestamp = int(messages[-1][0])

    def set_user_input_requested(self, requested: bool) -> None:
        if not self.message_db_conn:
            return
        self.message_db_conn.execute(
            "UPDATE status SET user_input_requested = ? WHERE id = 1",
            (1 if requested else 0,),
        )
        self.message_db_conn.commit()

    def _request_tool_approval_via_task_manager(self, tool_name: str, tool_kwargs: Dict[str, Any]) -> bool:
        """Relay approval requests through the task manager input channels."""
        serialized_args = json.dumps(tool_kwargs, default=str)
        prompt = (
            f"Tool '{tool_name}' requires approval before execution.\n"
            f"Arguments: {serialized_args}\n"
            "Approve? [y/N]: "
        )
        response = self.get_user_input(
            prompt,
            display_prompt_in_webui=self.use_webui,
        )
        approved = response.strip().lower() in {"y", "yes"}
        logger.info(
            "Tool '%s' approval %s by user via task manager.",
            tool_name,
            "granted" if approved else "denied",
        )
        return approved

    def run(self, *args, **kwargs) -> None:
        self.prerun_check()
        
    def display_command_help(self) -> str:            
        s = (
            "Below are supported commands. Note that not all commands are availble "
            "in your current environment. Refer to the input prompt for available commands.\n"
            "* `/exit`: exit the current loop\n"
            "* `/chat`: enter chat mode. The task manager will always ask for you input "
            "(instead of sending workflow-determined replies to the agent) when the "
            "agent finishes its response.\n"
            "* `/monitor <task description>`: enter monitoring mode. The agent will perform "
            "the described task periodically. Example: `/monitor check the content of status.txt `"
            "every 60 seconds`\n"
            "* `/subtask <task description>`: run a skill-driven subtask using the configured "
            "skill tools and coding tools.\n"
            "* `/return`: return to upper level task\n"
        )
        if self.use_webui:
            if self.message_db_conn:
                self.add_message_to_db({"role": "system", "content": s})
        else:
            print(s)
            if self.message_db_conn:
                self.add_message_to_db({"role": "system", "content": s})
        return s

    def get_manager_metadata_summary(self) -> str:
        llm_summary = repr(self.llm_config)
        llm_import_path = None
        if self.llm_config is not None:
            llm_class = self.llm_config.__class__
            llm_import_path = f"{llm_class.__module__}.{llm_class.__name__}"
        tool_info = []
        excluded_tools = {SkillTool, PythonCodingTool, BashCodingTool}
        for tool in self._collect_base_tools():
            if tool.__class__ in excluded_tools:
                continue
            tool_class = tool.__class__
            tool_info.append(
                {
                    "class_name": tool_class.__name__,
                    "import_path": f"{tool_class.__module__}.{tool_class.__name__}",
                }
            )
        payload = {
            "llm_config": llm_summary,
            "llm_config_import_path": llm_import_path,
            "tools": tool_info,
        }
        return json.dumps(payload, indent=2, default=str)

    def launch_task_manager(self, task_request: str) -> None:
        if self.agent is None:
            logger.warning("Cannot launch a skill task because no agent is configured.")
            system_message = generate_openai_message(
                content="Cannot launch a skill task because no agent is configured.",
                role="system",
            )
            self.update_message_history(system_message, update_context=True, update_full_history=True)
            print_message(system_message)
            return

        if not self.skill_catalog:
            logger.info("No skills discovered; skipping skill task launch.")
            system_message = generate_openai_message(
                content="No skills are available to run the subtask.",
                role="system",
            )
            self.update_message_history(system_message, update_context=True, update_full_history=True)
            print_message(system_message)
            return

        request_text = task_request.strip() or "(no additional description provided)"
        skill_catalog = [
            {
                "name": skill.name,
                "tool_name": skill.tool_name,
                "description": skill.description,
                "path": skill.path,
            }
            for skill in self.skill_catalog
        ]
        skill_catalog_json = json.dumps(skill_catalog, indent=2)
        skill_prompt = (
            "The user requested to launch a sub-task manager for a specified task. Create a Python "
            "script to set up the task manager, and execute it.\n"
            "The skill tools can be used to retrieved detailed documentation about each task manager. "
            "Select the proper task manager that fulfills the user's request, and call the skill tool; "
            "alternatively, you can also use the bash tool to look inside the skill directory according "
            "to the path of the skill tool and browse documentation files there.\n"
            "After reading the documentation, create a script to launch the task manager. "
            "If you lack critical information needed to set up the task manager, respond with 'NEED HUMAN' "
            "and ask a single clarification question.\n"
            "When the subtask is complete, respond with 'TERMINATE'.\n"
            f"Available skill tools:\n{skill_catalog_json}"
        )

        system_message = generate_openai_message(content=skill_prompt, role="system")
        self.update_message_history(system_message, update_context=True, update_full_history=True)
        metadata_message = generate_openai_message(
            content=(
                "Current task manager metadata (llm config + tools):\n"
                f"{self.get_manager_metadata_summary()}"
            ),
            role="system",
        )
        self.update_message_history(metadata_message, update_context=True, update_full_history=True)
        self.run_feedback_loop(
            initial_prompt=request_text,
            termination_behavior="return",
            allow_multiple_tool_calls=True,
        )

        self._sync_webui_user_input_last_timestamp()

    def run_conversation(
        self, 
        store_all_images_in_context: bool = True, 
        *args, **kwargs
    ) -> None:
        """Start a free-dtyle conversation with the assistant.

        Parameters
        ----------
        store_all_images_in_context : bool, optional
            Whether to store all images in the context. If False, only the image
            in the initial prompt, if any, is stored in the context. Keep this
            False to reduce the context size and save costs.
        """
        response = None
        while True:
            try:
                if response is None or (response is not None and not has_tool_call(response)):
                    message = self.get_user_input(
                        prompt=(
                            "Enter a message (/exit: exit; /return: return to upper level task; "
                            "/help: show command help): "
                        )
                    )
                    stripped_message = message.strip()
                    command, _, remainder = stripped_message.partition(" ")
                    command_lower = command.lower()

                    if command_lower == "/exit" and remainder == "":
                        break
                    elif command_lower == "/return" and remainder == "":
                        return
                    elif command_lower == "/monitor":
                        if len(remainder.strip()) == 0:
                            logger.info("Monitoring command requires a task description.")
                        else:
                            self.enter_monitoring_mode(remainder.strip())
                        continue
                    elif command_lower == "/subtask":
                        self.launch_task_manager(remainder.strip())
                        continue
                    elif command_lower == "/help" and remainder == "":
                        self.display_command_help()
                        continue
                
                    # Send message and get response
                    response, outgoing_message = self.agent.receive(
                        message, 
                        context=self.context, 
                        return_outgoing_message=True
                    )
                    # If WebUI is used, user input should come from WebUI which writes
                    # to the DB, so we don't update DB again.
                    self.update_message_history(
                        outgoing_message, 
                        update_context=True, 
                        update_full_history=True, 
                        update_db=not self.use_webui
                    )
                    self.update_message_history(response, update_context=True, update_full_history=True)
                
                # Handle tool calls
                tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
                tool_call_info_list = get_tool_call_info(response, index=None) if has_tool_call(response) else []
                for tool_response, tool_response_type in zip(tool_responses, tool_response_types):
                    print_message(tool_response)
                    self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if len(tool_responses) >= 1:
                    for index, (tool_response, tool_response_type) in enumerate(zip(tool_responses, tool_response_types)):
                        tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None

                        self._inject_skill_doc_messages_to_context(tool_response, tool_call_info)

                        # If the tool returns an image path, load the image and send it to 
                        # the assistant in a follow-up message as user.
                        if tool_response_type == ToolReturnType.IMAGE_PATH:
                            image_path = tool_response["content"]
                            image_message = generate_openai_message(
                                content="Here is the image the tool returned.",
                                image_path=image_path,
                                role="user",
                            )
                            self.update_message_history(
                                image_message, update_context=store_all_images_in_context, update_full_history=True
                            )
                    # Send tool responses stored in the context
                    response = self.agent.receive(
                        message=None, 
                        context=self.context, 
                        return_outgoing_message=False
                    )
                    self.update_message_history(response, update_context=True, update_full_history=True)
            except KeyboardInterrupt:
                self.context = complete_unresponded_tool_calls(self.context)
                response = generate_openai_message(
                    content="Workflow interrupted by keyboard interrupt. TERMINATE",
                    role="system"
                )
                self.update_message_history(
                    response, 
                    update_context=True, 
                    update_full_history=True
                )
                continue

    def enter_monitoring_mode(
        self,
        task_description: str,
    ):
        local_context = []
        parsing_prompt = (
            "Parse the following task description and return a JSON object with the following fields:\n"
            "- task_description: str\n"
            "- time_interval: float; time interval in seconds\n"
            "Example: given \"check the content of status.txt every 60 seconds\", the JSON object should be:\n"
            "{\n"
            "    \"task_description\": \"check the content of status.txt\",\n"
            "    \"time_interval\": 60\n"
            "}\n"
            "If enough information is provided, return the only the JSON object. Do not respond with anything else. "
            "Do not enclose the JSON object in triple backticks."
            "If any information is missing, ask the user for clarification.\n"
            f"Task description to be parsed: {task_description}\n"
        )
        
        while True:
            response, outgoing = self.agent.receive(
                message=parsing_prompt,
                context=local_context,
                return_outgoing_message=True,
            )
            self.update_message_history(outgoing, update_context=False, update_full_history=True)
            self.update_message_history(response, update_context=False, update_full_history=True)
            local_context.append(outgoing)
            local_context.append(response)
            try:
                parsed_task_description = json.loads(response["content"])
            except json.JSONDecodeError:
                parsing_prompt = self.get_user_input(
                    prompt=f"Failed to parse the task description. Please try again. {response['content']}",
                    display_prompt_in_webui=self.use_webui,
                )
                continue
            break
        
        self.run_monitoring(
            task_description=parsed_task_description["task_description"],
            time_interval=parsed_task_description["time_interval"],
        )

    def run_monitoring(
        self,
        task_description: str,
        time_interval: float,
        initial_prompt: Optional[str] = None,
    ):
        """Run a monitoring task.

        Parameters
        ----------
        task_description : str
            The task description.
        time_interval : float
            The time interval in seconds to run the task.
        initial_prompt : Optional[str], optional
            If provided, the default initial prompt will be overridden.
        """
        if initial_prompt is None:
            initial_prompt = (
                "You are given the following task needed to monitor the status of an experiment: "
                f"{task_description}\n"
                "Use proper trigger words in your response in the following scenarios:\n"
                "- You have checked all the statuses and everything is right - add \"TERMINATE\".\n"
                "- Something is wrong, but you have fixed it (if user instructed you to fix it) - add \"TERMINATE\".\n"
                "- Something is wrong, and you need immediate human input - add \"NEED HUMAN\".\n\n"
            )
        
        while True:
            try:
                self.run_feedback_loop(
                    initial_prompt=initial_prompt,
                    termination_behavior="return",
                )
                time.sleep(time_interval)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt detected. Terminating monitoring task.")
                self.add_message_to_db(
                    generate_openai_message(
                        content="Keyboard interrupt detected. Terminating monitoring task.",
                        role="system",
                    )
                )
                return
        
    def enforce_tool_call_sequence(
        self,
        expected_tool_call_sequence: list[str],
        tolerance: int = 0,
    ) -> None:
        """Check whether the current tool call is the one expected to be
        made after the last tool call as in the given sequence. If not,
        a warning is added to the context.

        Parameters
        ----------
        expected_tool_call_sequence : list[str]
            A list of tool names that are expected to be called in order.
        tolerance : int, optional
            The number of tool calls to tolerate in the expected tool call sequence.
            For example, if the actual tool call sequence is [X, B, C] while the
            expected tool call sequence is [B, C], the check will still pass if the
            tolerance is 1 because only [B, C] is used for matching. This is useful
            when the first several calls are only done once instead of iteratively.
        """
        if len(self.agent.tool_manager.tool_execution_history) <= 1:
            return
        
        n_actual_calls_to_fetch = min(
            len(self.agent.tool_manager.tool_execution_history), len(expected_tool_call_sequence)
        ) - tolerance
        
        if n_actual_calls_to_fetch <= 0:
            return
        actual_tool_call_sequence = [
            entry["tool_name"] for entry in self.agent.tool_manager.tool_execution_history[-n_actual_calls_to_fetch:]
        ]
        
        # Replicate the expected sequence so that it matches circularly.
        expanded_expected_tool_call_sequence = list(expected_tool_call_sequence) * 2
        
        # Match the actual tool call sequence with the expanded expected tool call sequence.
        for i in range(len(expanded_expected_tool_call_sequence) - len(actual_tool_call_sequence) + 1):
            if expanded_expected_tool_call_sequence[i:i+len(actual_tool_call_sequence)] == actual_tool_call_sequence:
                return
        self.context.append(
            generate_openai_message(
                content=(
                    f"The tool call sequence {actual_tool_call_sequence} is not as expected. "
                    f"Are you making the right tool calls in the right order? If this is intended "
                    f"to address an exception, ignore this message."
                ),
                role="user"
            )
        )

    def run_feedback_loop(
        self,
        initial_prompt: str,
        initial_image_path: Optional[str] = None,
        message_with_acquired_image: str = "Here is the image the tool returned.",
        max_rounds: int = 99,
        n_first_images_to_keep_in_context: Optional[int] = None,
        n_last_images_to_keep_in_context: Optional[int] = None,
        allow_non_image_tool_responses: bool = True,
        allow_multiple_tool_calls: bool = False,
        hook_functions: Optional[dict[str, Callable]] = None,
        expected_tool_call_sequence: Optional[list[str]] = None,
        expected_tool_call_sequence_tolerance: int = 0,
        termination_behavior: Literal["ask", "return"] = "ask",
        max_arounds_reached_behavior: Literal["return", "raise"] = "return",
        *args, **kwargs
    ) -> None:
        """Run an agent-involving feedback loop.
        
        The loop workflow is as follows:
        1. The agent is given an initial prompt, optionally with an image. 
        2. The agent makes a tool call; the return of the tool is 
           expected to be a path to an acquired image.
        3. The tool call is executed.
        4. The tool response is added to the context.
        5. The actual image is loaded and encoded.
        6. The tool response and image are sent to the agent.
        7. Go back to 2 and repeat until the agent responds with "TERMINATE".
        
        Each time the agent calls a tool, only one tool call is allowed. If multiple
        tool calls are made, the agent will be asked to redo the tool calls.
        
        Termination signals: "TERMINATE" and "NEED HUMAN". When "TERMINATE" is
        present, the function either returns or asks for user input depending on
        the setting of `termination_behavior`. When "NEED HUMAN" is present, the
        function always asks for user input.

        Parameters
        ----------
        initial_prompt : str
            The initial prompt for the agent.
        initial_image_path : str, optional
            The initial image path for the agent.
        message_with_acquired_image : str, optional
            The message to send to the agent along with the acquired image.
        max_rounds : int, optional
            The maximum number of rounds to run.
        n_first_images_to_keep, n_last_images_to_keep : int, optional
            The number of first and last images to keep in the context. If both of
            them are None, all images will be kept.
        allow_non_image_tool_responses : bool, optional
            If False, the agent will be asked to redo the tool call if it returns
            anything that is not an image path.
        allow_multiple_tool_calls: bool, optional
            If True, the agent will be allowed to make multiple tool calls in one
            response. If False, the agent will be asked to redo the tool call if it
            makes multiple tool calls.
        hook_functions : dict[str, Callable], optional
            A dictionary of hook functions to call at certain points in the loop.
            The keys specify the points where the hook functions are called, and
            the values are the callables. Allowed keys are:
            - `image_path_tool_response`: 
              - args: {"img_path": str}
              - return: list[Dict[str, Any]]: A list of messages that should be added 
                to the context.
              - Executed when the tool response is an image path, after the tool
                response is added to the context but before the image is loaded and
                sent to the agent. When this function is given, it **overrides** the
                routine that loads, encodes images and composes the image-containing
                follow-up message, so be sure to make the hook return a message that
                contains the loaded image if desired.
        expected_tool_call_sequence : list[str], optional
            A list of tool call names that are expected to be made in the loop.
            If provided, the function will check if the tool call sequence is
            as expected. A warning is issued to the agent if the sequence is 
            not as expected.
        expected_tool_call_sequence_tolerance : int, optional
            If given, the actual tool call sequence used for matching will be the
            last `len(expected_tool_call_sequence) - tolerance` calls. For example,
            if the actual tool call sequence is [X, B, C] while the expected tool
            call sequence is [B, C], the check will still pass if the tolerance is
            1 because only [B, C] is used for matching. This is useful when the
            first several calls are only done once instead of iteratively.
        termination_behavior : Literal["ask", "return"]
            Decides what to do when the agent sends termination signal ("TERMINATE")
            in the response. If "ask", the user will be asked to provide further
            instructions. If "return", the function will return directly.
        max_arounds_reached_behavior : Literal["return", "raise"], optional
            Decides what to do when the agent reaches the maximum number of
            rounds. If "return", the function will return directly. If "raise",
            the function will raise an error.
        """
        if termination_behavior not in ["ask", "return"]:
            raise ValueError("`termination_behavior` must be either 'ask' or 'return'.")
        
        hook_functions = hook_functions or {}
        round = 0
        image_path = None
        response, outgoing = self.agent.receive(
            initial_prompt,
            context=self.context,
            image_path=initial_image_path,
            return_outgoing_message=True
        )
        self.update_message_history(outgoing, update_context=True, update_full_history=True)
        self.update_message_history(response, update_context=True, update_full_history=True)
        while round < max_rounds:
            try:
                if "content" in response and response["content"] is not None:
                    if "TERMINATE" in response["content"] and termination_behavior == "return":
                        return
                    if (
                        ("TERMINATE" in response["content"] and termination_behavior == "ask") 
                        or "NEED HUMAN" in response["content"]
                    ):
                        message = self.get_user_input(
                            prompt=(
                                "Termination condition triggered. What to do next? "
                                "(`/exit`: exit; `/chat`: chat mode; `/help`: show command help): "
                            ),
                            display_prompt_in_webui=self.use_webui
                        )
                        if message.lower() == "/exit":
                            return
                        elif message.lower() == "/chat":
                            self.run_conversation(store_all_images_in_context=True)
                        elif message.lower() == "/help":
                            self.display_command_help()
                            continue
                        else:
                            response, outgoing = self.agent.receive(
                                message,
                                context=self.context,
                                image_path=None,
                                return_outgoing_message=True
                            )
                            self.update_message_history(outgoing, update_context=True, update_full_history=True)
                            self.update_message_history(response, update_context=True, update_full_history=True)
                            continue
                
                tool_responses, tool_response_types = self.agent.handle_tool_call(response, return_tool_return_types=True)
                tool_call_info_list = get_tool_call_info(response, index=None) if has_tool_call(response) else []
                
                # Add tool response to context regardless whether multiple tool calls are allowed or not, 
                # because otherwise it would throw an error.
                for tool_response, tool_response_type in zip(tool_responses, tool_response_types):
                    print_message(tool_response)
                    self.update_message_history(tool_response, update_context=True, update_full_history=True)
                
                if len(tool_responses) == 1 or allow_multiple_tool_calls:
                    for index, (tool_response, tool_response_type) in enumerate(zip(tool_responses, tool_response_types)):
                        tool_call_info = tool_call_info_list[index] if index < len(tool_call_info_list) else None

                        self._inject_skill_doc_messages_to_context(tool_response, tool_call_info)

                        if tool_response_type == ToolReturnType.IMAGE_PATH:
                            # If the tool returns an image path, load and encode the image,
                            # then compose a follow-up message with the image and add it to
                            # the context.
                            image_path = tool_response["content"]
                            if hook_functions.get("image_path_tool_response", None) is not None:
                                # Override the normal routine composing image-containing follow-up message
                                # with the hook function and the message it returns.
                                hook_messages = hook_functions["image_path_tool_response"](image_path)
                                for hook_message in hook_messages:
                                    self.update_message_history(hook_message, update_context=True, update_full_history=True)
                            else:
                                image_followup = generate_openai_message(
                                    content=message_with_acquired_image,
                                    image_path=image_path
                                )
                                self.update_message_history(image_followup, update_context=True, update_full_history=True)
                        else: 
                            if not allow_non_image_tool_responses:
                                msg = generate_openai_message(
                                    f"The tool should return an image path, but got {str(tool_response_type)}. "
                                    "Make sure you call the right tool correctly.",
                                )
                                self.update_message_history(msg, update_context=True, update_full_history=True)
                                
                    if expected_tool_call_sequence is not None:
                        self.enforce_tool_call_sequence(
                            expected_tool_call_sequence,
                            expected_tool_call_sequence_tolerance
                        )
                                
                    # Send the tool responses and any follow-ups for all tool calls.
                    response = self.agent.receive(
                        message=None,
                        image_path=None,
                        context=self.context,
                        return_outgoing_message=False
                    )
                    self.update_message_history(response, update_context=True, update_full_history=True)
                    
                elif len(tool_responses) > 1:
                    response, outgoing = self.agent.receive(
                        "There are more than one tool calls in your response. "
                        "Make sure you only make one call at a time. Please redo "
                        "your tool calls.", 
                        image_path=None,
                        context=self.context,
                        return_outgoing_message=True
                    )
                    self.update_message_history(outgoing, update_context=True, update_full_history=True)
                    self.update_message_history(response, update_context=True, update_full_history=True)
                else:
                    if "content" in response and response["content"] is not None and "NEED HUMAN" not in response["content"]:
                        response, outgoing = self.agent.receive(
                            "There is no tool call in the response. Make sure you call the tool correctly. "
                            "If you need human intervention, say \"NEED HUMAN\".",
                            image_path=None,
                            context=self.context,
                            return_outgoing_message=True
                        )
                        self.update_message_history(outgoing, update_context=True, update_full_history=True)
                        self.update_message_history(response, update_context=True, update_full_history=True)
                    else:
                        continue
                
                if n_last_images_to_keep_in_context is not None or n_first_images_to_keep_in_context is not None:
                    n_last_images_to_keep_in_context = n_last_images_to_keep_in_context if n_last_images_to_keep_in_context is not None else 0
                    n_first_images_to_keep_in_context = n_first_images_to_keep_in_context if n_first_images_to_keep_in_context is not None else 0
                    self.context = purge_context_images(
                        context=self.context,
                        keep_first_n=n_first_images_to_keep_in_context, 
                        keep_last_n=n_last_images_to_keep_in_context - 1,
                        keep_text=True
                    )
                round += 1
            except KeyboardInterrupt:
                self.context = complete_unresponded_tool_calls(self.context)
                response = generate_openai_message(
                    content="Workflow interrupted by keyboard interrupt. TERMINATE",
                    role="system"
                )
                self.update_message_history(
                    response, 
                    update_context=True, 
                    update_full_history=True
                )
                continue
        logger.warning(f"Maximum number of rounds ({max_rounds}) reached.")
        if max_arounds_reached_behavior == "raise":
            raise MaxRoundsReached()
        else:
            return
