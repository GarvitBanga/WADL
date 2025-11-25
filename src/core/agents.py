from dataclasses import dataclass
from time import time
from typing import List
from sqlalchemy.orm import Session
from src.db.models import AgentLog, Run

@dataclass
class AgentThought:
    agent_name: str
    action: str
    reasoning: str
    status: str
    timestamp: float

class AgentLogger:
    def __init__(self):
        self._buffer: List[AgentThought] = []

    def log(self, agent_name: str, action: str, reasoning: str, status: str):
        self._buffer.append(
            AgentThought(
                agent_name=agent_name,
                action=action,
                reasoning=reasoning,
                status=status,
                timestamp=time(),
            )
        )

    def flush_to_db(self, session: Session, run: Run):
        from datetime import datetime
        for t in self._buffer:
            session.add(
                AgentLog(
                    run_id=run.id,
                    agent_name=t.agent_name,
                    action=t.action,
                    reasoning=t.reasoning,
                    status=t.status,
                    timestamp=datetime.fromtimestamp(t.timestamp),
                )
            )
        session.commit()
        self._buffer.clear()

