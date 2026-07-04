# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Router component for IT support tickets.

The Router decides whether a ticket should be handled by the Support Resolution
Agent, the Human Escalation Agent, or a fallback path when neither is suitable.
It does not classify tickets into a support taxonomy in the minimum
implementation.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple

from .model_store import (
    NLI_ROUTER_MODEL_ID,
    is_router_model_downloaded,
    is_router_openvino_model_available,
    router_model_path,
    router_openvino_model_path,
)
from .schemas import RouterResult


LOW_CONFIDENCE_ESCALATION_THRESHOLD = 0.5
OPENVINO_DEFAULT_DEVICE = "AUTO"
OPENVINO_DEFAULT_MAX_LENGTH = 256


class NliRouterProtocol(Protocol):
    """Protocol for test doubles and local NLI Router implementations."""

    def route(self, ticket_text: str) -> RouterResult:
        """Return a binary Router decision for the supplied ticket."""


class NliMiniLmRouter:
    """Binary Router backed by cross-encoder/nli-MiniLM2-L6-H768."""

    SUPPORT_AGENT = "support_resolution_agent"
    ESCALATION_AGENT = "human_escalation_agent"
    LABEL_MAPPING = ("contradiction", "entailment", "neutral")
    ENTAILMENT_INDEX = 1
    SUPPORT_HYPOTHESIS = (
        "This IT support ticket should be handled automatically by the support resolution agent "
        "because it is a technical issue that can be answered from support knowledge."
    )
    ESCALATION_HYPOTHESIS = (
        "This IT support ticket should be sent to the human escalation agent because it involves "
        "safety risk, privileged access, policy approval, sensitive information, or unclear requirements."
    )

    def __init__(self, model_path: str | Path, confidence_threshold: float = LOW_CONFIDENCE_ESCALATION_THRESHOLD) -> None:
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._model = None

    def route(self, ticket_text: str) -> RouterResult:
        """Run NLI zero-shot routing over the two next-agent hypotheses."""
        support_score, escalation_score = self._predict_entailment_scores(ticket_text)
        return self._build_result(support_score, escalation_score)

    def _build_result(self, support_score: float, escalation_score: float) -> RouterResult:
        support_probability, escalation_probability = _softmax_pair(support_score, escalation_score)
        selected_probability = max(support_probability, escalation_probability)
        score_signals = self._score_signals(support_score, escalation_score)

        if selected_probability < self.confidence_threshold:
            return RouterResult(
                next_agent=self.ESCALATION_AGENT,
                confidence=float(selected_probability),
                reason="The NLI Router model had low separation between the two agent choices.",
                signals=score_signals + ["low nli confidence", "human review preferred"],
            )

        if escalation_probability > support_probability:
            return RouterResult(
                next_agent=self.ESCALATION_AGENT,
                confidence=float(escalation_probability),
                reason="The NLI Router model found stronger entailment for the human escalation agent.",
                signals=score_signals + ["nli escalation hypothesis selected"],
            )

        return RouterResult(
            next_agent=self.SUPPORT_AGENT,
            confidence=float(support_probability),
            reason="The NLI Router model found stronger entailment for the support resolution agent.",
            signals=score_signals + ["nli support hypothesis selected"],
        )

    def _score_signals(self, support_score: float, escalation_score: float) -> List[str]:
        return [
            f"nli model: {NLI_ROUTER_MODEL_ID}",
            f"support entailment={support_score:.3f}",
            f"escalation entailment={escalation_score:.3f}",
        ]

    def _predict_entailment_scores(self, ticket_text: str) -> Tuple[float, float]:
        model = self._load_model()
        pairs = [
            (ticket_text, self.SUPPORT_HYPOTHESIS),
            (ticket_text, self.ESCALATION_HYPOTHESIS),
        ]
        scores = model.predict(pairs)
        rows = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return self._extract_entailment_scores(rows)

    def _extract_entailment_scores(self, rows) -> Tuple[float, float]:
        if len(rows) != 2:
            raise RuntimeError(f"Expected two NLI score rows, received {len(rows)}")
        try:
            support_score = float(rows[0][self.ENTAILMENT_INDEX])
            escalation_score = float(rows[1][self.ENTAILMENT_INDEX])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Unexpected NLI score shape. Expected labels in order: "
                "contradiction, entailment, neutral."
            ) from exc
        return support_score, escalation_score

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Install `sentence-transformers` or run `pip install -r requirements.txt` "
                "before using the NLI Router."
            ) from exc
        self._model = CrossEncoder(str(self.model_path))
        return self._model


class OpenVinoNliMiniLmRouter(NliMiniLmRouter):
    """OpenVINO IR Router for Intel CPU/GPU/NPU accelerated inference."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        openvino_model_path: Optional[str | Path] = None,
        confidence_threshold: float = LOW_CONFIDENCE_ESCALATION_THRESHOLD,
        device: Optional[str] = None,
        max_length: int = OPENVINO_DEFAULT_MAX_LENGTH,
    ) -> None:
        super().__init__(model_path, confidence_threshold=confidence_threshold)
        self.openvino_model_path = Path(openvino_model_path) if openvino_model_path else router_openvino_model_path(model_path)
        self.device = (device or os.getenv("IT_SUPPORT_ROUTER_OPENVINO_DEVICE") or OPENVINO_DEFAULT_DEVICE).strip().upper()
        self.max_length = max_length
        self._tokenizer = None

    def _score_signals(self, support_score: float, escalation_score: float) -> List[str]:
        return super()._score_signals(support_score, escalation_score) + [
            "openvino router",
            f"device={self.device}",
            f"max_length={self.max_length}",
        ]

    def _predict_entailment_scores(self, ticket_text: str) -> Tuple[float, float]:
        tokenizer = self._load_tokenizer()
        compiled_model = self._load_model()
        encoded = tokenizer(
            [ticket_text, ticket_text],
            [self.SUPPORT_HYPOTHESIS, self.ESCALATION_HYPOTHESIS],
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        input_names = {input_node.any_name for input_node in compiled_model.inputs}
        inputs = {name: value for name, value in encoded.items() if name in input_names}
        if not inputs:
            raise RuntimeError(f"No tokenizer outputs matched OpenVINO model inputs: {sorted(input_names)}")
        result = compiled_model(inputs)
        output = compiled_model.outputs[0]
        rows = result[output].tolist()
        return self._extract_entailment_scores(rows)

    def _load_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Install `transformers` or run `pip install -r requirements.txt` "
                "before using the OpenVINO Router."
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model
        if not self.openvino_model_path.exists() or not self.openvino_model_path.with_suffix(".bin").exists():
            raise FileNotFoundError(f"OpenVINO Router model is not available at {self.openvino_model_path}")
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise RuntimeError(
                "Install `openvino` or run `pip install -r requirements.txt` "
                "before using the OpenVINO Router."
            ) from exc
        core = ov.Core()
        self._model = core.compile_model(str(self.openvino_model_path), self.device)
        return self._model


class Router:
    """Choose the next support agent for a ticket."""

    SUPPORT_AGENT = "support_resolution_agent"
    ESCALATION_AGENT = "human_escalation_agent"
    FALLBACK_AGENT = "fallback_agent"
    DOWNLOAD_COMMAND = "python -m router_agent.src.download_router_model"

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        model_path: Optional[str | Path] = None,
        nli_router: Optional[NliRouterProtocol] = None,
        openvino_router: Optional[NliRouterProtocol] = None,
        nli_confidence_threshold: float = LOW_CONFIDENCE_ESCALATION_THRESHOLD,
    ) -> None:
        selected_backend = backend or os.getenv("IT_SUPPORT_ROUTER_BACKEND") or "auto"
        self.backend = selected_backend.strip().lower()
        if self.backend not in {"auto", "openvino", "nli", "heuristic"}:
            raise ValueError("Router backend must be one of: auto, openvino, nli, heuristic")

        self.model_path = router_model_path(model_path)
        self.nli_confidence_threshold = nli_confidence_threshold
        self._nli_router = nli_router
        self._openvino_router = openvino_router
        self.safety_patterns: List[Tuple[str, str]] = [
            (r"\b(swollen|smoke|burning|sparks|fire|electric shock|overheating)\b", "safety risk"),
        ]
        self.unsafe_patterns: List[Tuple[str, str]] = [
            (
                r"\b(bypass mfa|crack password|steal credentials|credential theft|phish|phishing|keylogger|malware|ransomware|ddos|hack)\b",
                "unsafe misuse request",
            ),
        ]
        self.approval_patterns: List[Tuple[str, str]] = [
            (
                r"\b(admin rights|administrator|elevated privileges|sudo|root access|install software|software install)\b",
                "privilege request",
            ),
            (r"\b(purchase|procurement|invoice|expense|reimbursement|budget approval)\b", "approval workflow"),
            (r"\b(hr|benefits|human resources|leave request|vacation|payroll)\b", "sensitive HR request"),
            (r"\b(policy exception|security exception|compliance|legal|audit)\b", "policy-sensitive request"),
        ]
        self.technical_patterns: List[Tuple[str, str]] = [
            (r"\b(error|failed|failure|not working|troubleshoot|issue|problem)\b", "technical support request"),
            (r"\b(vpn|login|password|mfa|authentication|connect|connection|install|configure)\b", "answerable technical issue"),
        ]
        self.conversational_patterns: List[Tuple[str, str]] = [
            (r"^\s*(hi|hello|hey|yo|good morning|good afternoon|good evening)\b[\s.!?]*$", "greeting only"),
            (r"^\s*(thanks|thank you|thx|ok|okay|cool)\b[\s.!?]*$", "acknowledgement only"),
            (r"^\s*(test|testing|ping)\b[\s.!?]*$", "non-ticket test message"),
        ]
        self.out_of_scope_patterns: List[Tuple[str, str]] = [
            (
                r"\b(weather|forecast|temperature|sports|football|soccer|basketball|recipe|cooking|movie recommendation|translate|translation|essay|poem|horoscope)\b",
                "out-of-scope request",
            ),
        ]

    def route_ticket(self, ticket_text: str) -> RouterResult:
        """Select the next support agent for a ticket."""
        text = ticket_text.strip().lower()
        guardrail_result = self._guardrail_route(text)
        if guardrail_result is not None:
            return guardrail_result

        if self.backend == "heuristic":
            return self._apply_confidence_policy(self._heuristic_route(text))

        if self.backend == "nli":
            return self._route_with_nli_or_raise(ticket_text, text)

        if self.backend == "openvino":
            return self._route_with_openvino_then_nli(ticket_text, text)

        return self._route_auto(ticket_text, text)

    def _route_auto(self, ticket_text: str, normalized_text: str) -> RouterResult:
        openvino_signal: Optional[str] = None
        if is_router_openvino_model_available(self.model_path):
            try:
                return self._route_with_openvino(ticket_text, normalized_text)
            except Exception as exc:
                openvino_signal = f"openvino router fallback: {exc}"
        else:
            openvino_signal = "openvino model missing; safetensors fallback"

        try:
            result = self._route_with_nli_or_raise(ticket_text, normalized_text)
            self._append_signal(result, openvino_signal)
            return result
        except Exception as exc:
            result = self._heuristic_route(normalized_text)
            self._append_signal(result, openvino_signal)
            result.signals.append(f"nli router fallback: {exc}")
            return self._apply_confidence_policy(result)

    def _route_with_openvino_then_nli(self, ticket_text: str, normalized_text: str) -> RouterResult:
        try:
            return self._route_with_openvino(ticket_text, normalized_text)
        except Exception as openvino_exc:
            try:
                result = self._route_with_nli_or_raise(ticket_text, normalized_text)
                result.signals.append(f"openvino router fallback: {openvino_exc}")
                return result
            except Exception as nli_exc:
                raise RuntimeError(
                    "OpenVINO Router is unavailable and the safetensors fallback could not be loaded. "
                    "Create the OpenVINO IR with `notebooks/convert_router_to_openvino_ir.ipynb` or "
                    f"download the base model with `{self.DOWNLOAD_COMMAND}`."
                ) from nli_exc

    def _route_with_openvino(self, ticket_text: str, normalized_text: str) -> RouterResult:
        result = self._get_openvino_router().route(ticket_text.strip())
        result = self._reconcile_model_result_with_heuristics(result, normalized_text)
        self._add_heuristic_context_signals(result, normalized_text)
        return self._apply_confidence_policy(result)

    def _route_with_nli_or_raise(self, ticket_text: str, normalized_text: str) -> RouterResult:
        try:
            result = self._get_nli_router().route(ticket_text.strip())
            result = self._reconcile_model_result_with_heuristics(result, normalized_text)
            self._add_heuristic_context_signals(result, normalized_text)
            return self._apply_confidence_policy(result)
        except Exception as exc:
            raise RuntimeError(
                "NLI Router is unavailable. Download the model with "
                f"`{self.DOWNLOAD_COMMAND}` and install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

    def _apply_confidence_policy(self, result: RouterResult) -> RouterResult:
        """Escalate automatic support routing when confidence is below threshold."""
        if result.next_agent != self.SUPPORT_AGENT:
            return result
        if result.confidence >= self.nli_confidence_threshold:
            return result

        signals = list(result.signals)
        for signal in ("low router confidence", "human review preferred"):
            if signal not in signals:
                signals.append(signal)
        return RouterResult(
            next_agent=self.ESCALATION_AGENT,
            confidence=result.confidence,
            reason=(
                "The Router confidence for automatic support resolution is below "
                f"{self.nli_confidence_threshold:.2f}, so the ticket should be reviewed by human support."
            ),
            signals=signals,
        )

    def _get_openvino_router(self) -> NliRouterProtocol:
        if self._openvino_router is not None:
            return self._openvino_router
        if not is_router_openvino_model_available(self.model_path):
            raise FileNotFoundError(f"OpenVINO Router model is not available at {router_openvino_model_path(self.model_path)}")
        self._openvino_router = OpenVinoNliMiniLmRouter(
            self.model_path,
            confidence_threshold=self.nli_confidence_threshold,
        )
        return self._openvino_router

    def _get_nli_router(self) -> NliRouterProtocol:
        if self._nli_router is not None:
            return self._nli_router
        if not is_router_model_downloaded(self.model_path):
            raise FileNotFoundError(f"Router model is not available at {self.model_path}")
        self._nli_router = NliMiniLmRouter(
            self.model_path,
            confidence_threshold=self.nli_confidence_threshold,
        )
        return self._nli_router

    def _guardrail_route(self, text: str) -> Optional[RouterResult]:
        if not text:
            return self._fallback_result(
                confidence=0.98,
                reason="The request is empty and cannot be processed by this workflow.",
                signals=["empty request", "clarification needed", "router guardrail"],
            )

        conversational_signals = self._matching_signals(text, self.conversational_patterns)
        if conversational_signals:
            return self._fallback_result(
                confidence=0.97,
                reason="The message is conversational and not an actionable IT support request.",
                signals=conversational_signals + ["router guardrail"],
            )

        unsafe_signals = self._matching_signals(text, self.unsafe_patterns)
        if unsafe_signals:
            return self._fallback_result(
                confidence=0.99,
                reason="The request is unsafe and should not be processed by this IT support workflow.",
                signals=unsafe_signals + ["router guardrail"],
            )

        safety_signals = self._matching_signals(text, self.safety_patterns)
        if safety_signals:
            return RouterResult(
                next_agent=self.ESCALATION_AGENT,
                confidence=0.95,
                reason="The request mentions a potential safety risk and should be reviewed by a human.",
                signals=safety_signals + ["router guardrail"],
            )

        approval_signals = self._matching_signals(text, self.approval_patterns)
        if approval_signals:
            return RouterResult(
                next_agent=self.ESCALATION_AGENT,
                confidence=0.88,
                reason="The request may require approval, privileged access, or policy review.",
                signals=approval_signals + ["router guardrail"],
            )

        out_of_scope_signals = self._matching_signals(text, self.out_of_scope_patterns)
        if out_of_scope_signals:
            return self._fallback_result(
                confidence=0.96,
                reason="The message is outside the scope of the IT support workflow.",
                signals=out_of_scope_signals + ["router guardrail"],
            )

        if self._is_unclear(text):
            return self._fallback_result(
                confidence=0.92,
                reason="The request is too unclear to route to support resolution or human escalation.",
                signals=["unclear request", "clarification needed", "router guardrail"],
            )
        return None

    def _fallback_result(self, *, confidence: float, reason: str, signals: List[str]) -> RouterResult:
        return RouterResult(
            next_agent=self.FALLBACK_AGENT,
            confidence=confidence,
            reason=reason,
            signals=signals,
        )

    def _heuristic_route(self, text: str) -> RouterResult:
        guardrail_result = self._guardrail_route(text)
        if guardrail_result is not None:
            return guardrail_result

        technical_signals = self._matching_signals(text, self.technical_patterns)
        if technical_signals:
            return RouterResult(
                next_agent=self.SUPPORT_AGENT,
                confidence=0.82,
                reason="The request appears safe and answerable with support knowledge.",
                signals=technical_signals + ["no escalation signal detected", "heuristic router"],
            )

        return RouterResult(
            next_agent=self.SUPPORT_AGENT,
            confidence=0.64,
            reason="No escalation signal was detected, so the request can start with support resolution.",
            signals=["general support request", "no escalation signal detected", "heuristic router"],
        )

    def _reconcile_model_result_with_heuristics(self, result: RouterResult, text: str) -> RouterResult:
        """Prefer deterministic support signals over weak model escalation.

        Guardrails for safety, privilege, policy, unsafe, unclear, and out-of-scope
        requests run before model routing. If a model still escalates a normal
        technical ticket, keep the workflow on the RAG support path.
        """
        if result.next_agent != self.ESCALATION_AGENT:
            return result

        technical_signals = self._matching_signals(text, self.technical_patterns)
        if not technical_signals:
            return result

        signals = list(result.signals)
        for signal in technical_signals + ["model escalation overridden by technical support signal"]:
            if signal not in signals:
                signals.append(signal)

        return RouterResult(
            next_agent=self.SUPPORT_AGENT,
            confidence=max(result.confidence, 0.82),
            reason=(
                "The request has deterministic technical-support signals and no escalation guardrail matched, "
                "so it should be handled by the support resolution agent."
            ),
            signals=signals,
        )

    def _add_heuristic_context_signals(self, result: RouterResult, text: str) -> None:
        context_signals: List[str] = []
        if result.next_agent == self.SUPPORT_AGENT:
            context_signals = self._matching_signals(text, self.technical_patterns)
            if not context_signals:
                context_signals = ["no escalation signal detected"]
        for signal in context_signals:
            if signal not in result.signals:
                result.signals.append(signal)

    def _append_signal(self, result: RouterResult, signal: Optional[str]) -> None:
        if signal and signal not in result.signals:
            result.signals.append(signal)

    def _matching_signals(self, text: str, patterns: Sequence[Tuple[str, str]]) -> List[str]:
        return [signal for pattern, signal in patterns if re.search(pattern, text)]

    def _is_unclear(self, text: str) -> bool:
        words = re.findall(r"\w+", text)
        if len(words) < 3:
            return True
        vague_only = {"help", "urgent", "issue", "problem", "broken", "please"}
        return len(words) <= 5 and all(word in vague_only for word in words)


def _softmax_pair(first: float, second: float) -> Tuple[float, float]:
    max_score = max(first, second)
    first_exp = math.exp(first - max_score)
    second_exp = math.exp(second - max_score)
    denominator = first_exp + second_exp
    return first_exp / denominator, second_exp / denominator
