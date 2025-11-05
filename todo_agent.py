from __future__ import annotations
import re
from typing import List, Optional
import logging

from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.core.config import LLMConfig, AgentConfig
from openhands.events.action import MessageAction
from openhands.llm.llm import LLM
from openhands.events.observation import Observation
from openhands.events.event import Event
from openhands.events.observation import FileReadObservation, CmdOutputObservation
from openhands.events.action import CmdRunAction, FileReadAction
from openhands.events.event import Event
import os

logger = logging.getLogger(__name__)


class TodoCodeActAgent(Agent):
    TARGET_TASK_FILE = "/instruction/task.md"

    def __init__(self, llm_config: LLMConfig, config: AgentConfig | None = None):
        self.llm = LLM(llm_config)
        super().__init__(llm=self.llm, config=config or AgentConfig())
        self.codeact = CodeActAgent(llm=self.llm, config=config)

        self.full_task: Optional[str] = None
        self.todo: List[str] = []
        self.current_step_index: int = 0
        self.state: str = "WAITING_FOR_TASK"  # "WAITING_FOR_TASK" or "EXECUTING"

    def reset(self):
        super().reset()
        self.full_task = None
        self.todo = []
        self.current_step_index = 0
        self.state = "WAITING_FOR_TASK"
        self.codeact.reset()

    def _check_if_task_file_read(self, state) -> Optional[str]:
        """
        ONLY check for MessageAction with extras.path == /instruction/task.md.
        Return raw content as-is — no cleaning, no parsing.
        """
        target_path = "/instruction/task.md"  # exact match, no normpath needed if fixed

        for event in reversed(state.history):
            if not isinstance(event, MessageAction):
                continue
            if getattr(event, 'extras', {}).get('path') == target_path:
                return event.content  # return exactly as received
        return None

    def _generate_todo(self, task: str) -> List[str]:
        # [Same as before — your existing implementation]
        prompt = (
            "You are a helpful assistant acting as a planner. "
            "Break down the following task into clear, ordered, and executable steps.\n"
            "Each step should be concise, start with a verb, if the task contain the entity such as name, application, link etc., plese also privided in each step.\n"
            "Return only the steps, one per line, without numbering or bullet symbols.\n\n"
            f"Task: {task}\n\n"
            "Steps:"
        )
        try:
            response = self.llm.completion(messages=[{'role': 'user', 'content': prompt}])
            content = response.choices[0].message.content.strip()
            steps = [
                line.strip()
                for line in content.splitlines()
                if len(line.strip()) > 2 and not re.match(r'^\s*[\d\-\*•]\s*', line)
            ]
            steps = [s for s in steps if s.lower() not in ['steps:', 'step:']]
            return steps[:10]
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}. Using fallback.")
            return ["Understand the task", "Implement solution", "Test and validate"]

    def step(self, state) -> MessageAction:
        # === Phase 1: Wait until /instruction/task.md is read ===
        if self.state == "WAITING_FOR_TASK":
            # Check if task file has just been read
            task_content = self._check_if_task_file_read(state)
            if task_content and task_content.strip():
                self.full_task = task_content.strip()
                self.todo = self._generate_todo(self.full_task)
                self.current_step_index = 0
                self.state = "EXECUTING"
                logger.info(f"[TodoCodeActAgent] Task file read. Generated {len(self.todo)} steps.")

                if self.todo:
                    first_step = self.todo[0]
                    plan_msg = (
                        f"✅ Read task from {self.TARGET_TASK_FILE}:\n```\n{self.full_task}\n```\n\n"
                        f"📋 Plan ({len(self.todo)} steps):\n" +
                        "\n".join(f"{i+1}. {s}" for i, s in enumerate(self.todo)) +
                        f"\n\n➡️ Starting step 1: {first_step}"
                    )
                    return MessageAction(content=plan_msg, source='agent', keep_details=True)
                else:
                    return MessageAction(content="Task is empty or planning failed.", source='agent')

            # Otherwise, let CodeAct proceed normally (e.g., it will run `cat /instruction/task.md`)
            return self.codeact.step(state)

        # === Phase 2: Execute planned steps ===
        elif self.state == "EXECUTING":
            if not self.todo or self.current_step_index >= len(self.todo):
                return MessageAction(content="🎉 All steps completed successfully.", source='agent')

            current_step = self.todo[self.current_step_index]
            prefix_msg = (
                f"[Planner] Step {self.current_step_index + 1}/{len(self.todo)}:\n{current_step}\n\n"
                "Use tools to make progress. When done, say exactly: 'STEP COMPLETED'."
            )

            # Inject planning context
            extended_history = [
                MessageAction(content=prefix_msg, source='user', keep_details=True)
            ] + list(state.history)

            # Reconstruct state with new history
            new_state = type(state)(**{**state.__dict__, 'history': extended_history})

            action = self.codeact.step(new_state)

            # Check completion signal
            if isinstance(action, MessageAction) and 'STEP COMPLETED' in action.content.strip():
                completed = self.todo[self.current_step_index]
                self.current_step_index += 1
                logger.info(f"[TodoCodeActAgent] Completed step: {completed}")

                if self.current_step_index < len(self.todo):
                    next_step = self.todo[self.current_step_index]
                    msg = f"✅ Completed: {completed}\n\n➡️ Next: {next_step}"
                    return MessageAction(content=msg, source='agent', keep_details=True)
                else:
                    return MessageAction(content="🎉 Task completed successfully.", source='agent')

            return action

        else:
            return MessageAction(content="Agent in unknown state.", source='agent')
