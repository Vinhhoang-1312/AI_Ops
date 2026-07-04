# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Seed Langfuse evaluation datasets for the IT support workflow."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.env import load_project_env


DEFAULT_DATASET_NAME = "it-support/end-to-end"


SEED_ITEMS: list[dict[str, Any]] = [
    {
        "ticket_text": "I cannot access the VPN from home. MFA succeeds, but the connection still fails.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["VPN", "network", "MFA", "credentials"],
        },
        "metadata": {"case_type": "access_support", "difficulty": "easy"},
    },
    {
        "ticket_text": "How do I reset my password after too many failed login attempts?",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["password", "reset", "account", "login"],
        },
        "metadata": {"case_type": "account_recovery", "difficulty": "easy"},
    },
    {
        "ticket_text": "My laptop battery is swollen and hot.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["human", "support", "battery", "safety"],
        },
        "metadata": {"case_type": "hardware_safety", "difficulty": "easy"},
    },
    {
        "ticket_text": "I need admin rights on production immediately.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["human", "approval", "review", "support"],
        },
        "metadata": {"case_type": "privileged_access", "difficulty": "easy"},
    },
    {
        "ticket_text": "The printer status remains pending after I restarted my laptop.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["printer", "queue", "restart", "driver"],
        },
        "metadata": {"case_type": "device_support", "difficulty": "medium"},
    },
    {
        "ticket_text": "Can you approve a software purchase for my department?",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["approval", "human", "procurement", "review"],
        },
        "metadata": {"case_type": "approval_workflow", "difficulty": "medium"},
    },
    {
        "ticket_text": "My email client cannot connect after the latest update.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["email", "update", "connection", "settings"],
        },
        "metadata": {"case_type": "email_support", "difficulty": "medium"},
    },
    {
        "ticket_text": "The screen flickers and I smell burning plastic near my laptop charger.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["human", "safety", "charger", "support"],
        },
        "metadata": {"case_type": "hardware_safety", "difficulty": "medium"},
    },
    {
        "ticket_text": "I cannot install Docker Desktop because the installer says permissions are required.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["approval", "permissions", "support", "review"],
        },
        "metadata": {"case_type": "software_install_approval", "difficulty": "medium"},
    },
    {
        "ticket_text": "My browser says the internal support portal certificate is invalid.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["certificate", "browser", "portal", "time"],
        },
        "metadata": {"case_type": "browser_support", "difficulty": "medium"},
    },
    {
        "ticket_text": "I need help configuring VPN on a new phone for work travel.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["VPN", "phone", "MFA", "configuration"],
        },
        "metadata": {"case_type": "mobile_access", "difficulty": "medium"},
    },
    {
        "ticket_text": "Payroll says my benefit information looks wrong. Can IT fix it?",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["human", "HR", "sensitive", "support"],
        },
        "metadata": {"case_type": "sensitive_hr", "difficulty": "medium"},
    },
    {
        "ticket_text": "My monitor is not detected after I connected the docking station.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["monitor", "dock", "cable", "display"],
        },
        "metadata": {"case_type": "hardware_support", "difficulty": "medium"},
    },
    {
        "ticket_text": "I need a policy exception to share customer data with an external vendor.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["policy", "approval", "human", "review"],
        },
        "metadata": {"case_type": "policy_sensitive", "difficulty": "hard"},
    },
    {
        "ticket_text": "The corporate Wi-Fi keeps disconnecting every few minutes in the office.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["Wi-Fi", "network", "disconnect", "adapter"],
        },
        "metadata": {"case_type": "network_support", "difficulty": "medium"},
    },
    {
        "ticket_text": "Can you bypass MFA for me because my phone is broken?",
        "expected_output": {
            "next_agent": "fallback_agent",
            "must_include_any": ["process", "support", "cannot", "help"],
        },
        "metadata": {"case_type": "unsafe_misuse", "difficulty": "hard"},
    },
    {
        "ticket_text": "The AppScan Source silent install fails on my RHEL machine.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["install", "RHEL", "error", "logs"],
        },
        "metadata": {"case_type": "techqa_install", "difficulty": "hard"},
    },
    {
        "ticket_text": "I need root access to the database server for a production migration tonight.",
        "expected_output": {
            "next_agent": "human_escalation_agent",
            "must_include_any": ["approval", "human", "privileged", "review"],
        },
        "metadata": {"case_type": "privileged_access", "difficulty": "hard"},
    },
    {
        "ticket_text": "My Teams calls have no microphone input, but the mic works in Windows settings.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["microphone", "Teams", "permissions", "audio"],
        },
        "metadata": {"case_type": "collaboration_tools", "difficulty": "medium"},
    },
    {
        "ticket_text": "The backup job for our shared drive failed last night with an authentication error.",
        "expected_output": {
            "next_agent": "support_resolution_agent",
            "must_include_any": ["backup", "authentication", "credentials", "logs"],
        },
        "metadata": {"case_type": "storage_backup", "difficulty": "hard"},
    },
]


def _configure_console() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def dataset_item_id(dataset_name: str, ticket_text: str) -> str:
    digest = hashlib.sha256(f"{dataset_name}:{ticket_text}".encode("utf-8")).hexdigest()[:24]
    return f"it-support-{digest}"


def _dataset_exists(langfuse, dataset_name: str) -> bool:
    try:
        langfuse.get_dataset(dataset_name)
        return True
    except Exception:
        return False


def seed_dataset(
    langfuse,
    *,
    dataset_name: str,
    items: list[dict[str, Any]],
    create_only: bool = False,
) -> int:
    if not _dataset_exists(langfuse, dataset_name):
        langfuse.create_dataset(
            name=dataset_name,
            description="End-to-end IT support evaluation cases for router, retrieval, and response quality.",
            metadata={"app": "agentic-rag-based-it-support", "seeded_by": "app.seed_langfuse_dataset"},
        )

    created = 0
    for item in items:
        ticket_text = str(item["ticket_text"]).strip()
        item_id = dataset_item_id(dataset_name, ticket_text)
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=item_id,
                input={"ticket_text": ticket_text},
                expected_output=item["expected_output"],
                metadata=item.get("metadata") or {},
            )
            created += 1
        except Exception:
            if create_only:
                continue
            raise

    langfuse.flush()
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed the Langfuse IT support evaluation dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Langfuse dataset name to create or populate")
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Skip existing duplicate seed items instead of failing.",
    )
    return parser


def main() -> None:
    _configure_console()
    args = build_parser().parse_args()
    load_project_env(PROJECT_ROOT)

    from langfuse import get_client

    langfuse = get_client()
    created = seed_dataset(
        langfuse,
        dataset_name=args.dataset,
        items=SEED_ITEMS,
        create_only=args.create_only,
    )
    print(f"Seeded {created} item(s) into Langfuse dataset '{args.dataset}'.")


if __name__ == "__main__":
    main()
