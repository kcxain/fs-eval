from typing import *
from dataclasses import dataclass, field


@dataclass
class EvalContent:
    error_msg: str
    ncu_profile: Dict[str, str] | None = None
    torch_profile: Dict[str, str] | None = None
    torch_time: float | None = None
    cuda_time: float | None = None
    speedup: Dict[str, str] | None = None


@dataclass
class Refine:
    ops_name: str = None
    code: str = None
    formated: bool = False
    compiled: bool = False
    passed: bool = False
    compile_msg: str | None = None
    eval_msg: EvalContent | None = None
    rag: Dict[str, str] | None = None

    def __post_init__(self):
        if not self.compiled:
            self.passed = False
        if not self.formated:
            self.compiled = False
            self.passed = False
        if self.passed:
            self.compiled = True
            self.formated = True
        if self.compiled:
            self.formated = True

    def validate(self) -> bool:
        if not self.compiled:
            assert self.compile_msg is not None, (
                "Compile message must be provided if compilation failed."
            )
        if self.passed:
            assert self.eval_msg is not None, (
                "Eval message must be provided if the code passed."
            )
        return True


@dataclass
class MultiRefine:
    max_turns: int = 5
    cur_turn: int = 0
    final_passed: bool = False
    final_compiled: bool = False
    final_speedup: float = -1
    refines: List[Refine] = field(default_factory=list)

    def update(self, refine: Refine):
        self.refines.append(refine)
        speedups = []
        for r in self.refines:
            if r.passed:
                self.final_passed = True
            if r.compiled:
                self.final_compiled = True
        if self.final_passed:
            for r in self.refines:
                if r.passed:
                    speedups.append(r.eval_msg.speedup)
        if speedups:
            self.final_speedup = max(speedups)
        self.cur_turn = len(self.refines)
        return self
