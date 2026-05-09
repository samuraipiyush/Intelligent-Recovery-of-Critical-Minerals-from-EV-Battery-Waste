"""
LLM agent with tool use.

Set ONE of these env vars before running:
  ANTHROPIC_API_KEY=sk-ant-...          → uses Claude (paid)
  GROQ_API_KEY=gsk_...                  → uses Groq/Llama (free tier)

Override provider explicitly:
  LLM_PROVIDER=groq   or   LLM_PROVIDER=anthropic
"""

import json
import os

os.environ.setdefault("GROQ_API_KEY", "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")
os.environ.setdefault("LLM_PROVIDER", "groq")

from models.composition import compute_composition
from models.recovery import compute_recovery
from models.economics import compute_revenue, compute_profit, prices as base_prices
from models.process import select_process

# ── provider detection ────────────────────────────────────────────────────────
_provider = os.environ.get("LLM_PROVIDER", "").lower()
if not _provider:
    if os.environ.get("GROQ_API_KEY"):
        _provider = "groq"
    elif os.environ.get("ANTHROPIC_API_KEY"):
        _provider = "anthropic"
    else:
        _provider = "anthropic"  # will fail at call time with a clear error

# ── tool definitions ──────────────────────────────────────────────────────────
# Anthropic format (used directly)
_TOOLS_ANTHROPIC = [
    {
        "name": "compute_recovery",
        "description": (
            "Compute material composition and recovered metal quantities for a given "
            "battery type and mass. Returns composition (kg), recovered metals (kg), "
            "and process recommendation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "battery_type": {"type": "string", "enum": ["NMC", "LFP", "LCO"]},
                "mass_kg": {"type": "number", "description": "Total battery mass in kg"},
            },
            "required": ["battery_type", "mass_kg"],
        },
    },
    {
        "name": "cost_analysis",
        "description": "Perform techno-economic analysis. Returns revenue breakdown and profit.",
        "input_schema": {
            "type": "object",
            "properties": {
                "battery_type": {"type": "string", "enum": ["NMC", "LFP", "LCO"]},
                "mass_kg": {"type": "number"},
                "processing_cost": {"type": "number", "description": "Processing cost in INR (₹)"},
            },
            "required": ["battery_type", "mass_kg", "processing_cost"],
        },
    },
    {
        "name": "process_selection",
        "description": "Recommend Hydrometallurgy or Pyrometallurgy for a battery type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "battery_type": {"type": "string", "enum": ["NMC", "LFP", "LCO"]},
            },
            "required": ["battery_type"],
        },
    },
    {
        "name": "sensitivity_analysis",
        "description": "Show how profit changes when a metal's price changes by a given percentage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "battery_type": {"type": "string", "enum": ["NMC", "LFP", "LCO"]},
                "mass_kg": {"type": "number"},
                "processing_cost": {"type": "number"},
                "metal": {"type": "string", "description": "Metal symbol, e.g. Li, Co, Ni"},
                "price_change_pct": {"type": "number", "description": "% change, e.g. -20"},
            },
            "required": ["battery_type", "mass_kg", "processing_cost", "metal", "price_change_pct"],
        },
    },
]

# OpenAI / Groq format (wraps same schemas inside "function")
_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in _TOOLS_ANTHROPIC
]

SYSTEM_PROMPT = (
    "You are an expert assistant for an EV battery recycling platform. "
    "You help users understand recovery of critical minerals from lithium-ion batteries (NMC, LFP, LCO). "
    "When the user asks a question that requires computation, call the appropriate tool. "
    "After getting tool results, explain them clearly — include specific numbers, units (kg, ₹), "
    "and actionable recommendations. Keep responses concise and factual. "
    "Available metals: Li, Co, Ni, Mn, Fe, P. Prices are in ₹/kg."
)


# ── backend tool execution ────────────────────────────────────────────────────
def _run_tool(name: str, inputs: dict) -> dict:
    if name == "compute_recovery":
        comp = compute_composition(inputs["battery_type"], inputs["mass_kg"])
        recovered = compute_recovery(comp)
        process, reason = select_process(inputs["battery_type"], recovered)
        return {"composition_kg": comp, "recovered_kg": recovered,
                "process": process, "process_reason": reason}

    if name == "cost_analysis":
        comp = compute_composition(inputs["battery_type"], inputs["mass_kg"])
        recovered = compute_recovery(comp)
        revenue, breakdown = compute_revenue(recovered)
        profit = compute_profit(revenue, inputs["processing_cost"])
        return {"revenue_breakdown_inr": breakdown, "total_revenue_inr": revenue,
                "processing_cost_inr": inputs["processing_cost"], "profit_inr": profit}

    if name == "process_selection":
        comp = compute_composition(inputs["battery_type"], 1.0)
        recovered = compute_recovery(comp)
        process, reason = select_process(inputs["battery_type"], recovered)
        return {"process": process, "reason": reason}

    if name == "sensitivity_analysis":
        comp = compute_composition(inputs["battery_type"], inputs["mass_kg"])
        recovered = compute_recovery(comp)
        revenue_base, _ = compute_revenue(recovered)
        profit_base = compute_profit(revenue_base, inputs["processing_cost"])

        metal = inputs["metal"]
        factor = 1 + inputs["price_change_pct"] / 100
        perturbed = dict(base_prices)
        if metal in perturbed:
            perturbed[metal] *= factor

        revenue_new = sum(recovered.get(m, 0) * perturbed.get(m, 0) for m in recovered)
        profit_new = compute_profit(revenue_new, inputs["processing_cost"])
        return {
            "baseline_profit_inr": profit_base,
            "new_profit_inr": profit_new,
            "profit_change_inr": profit_new - profit_base,
            "profit_change_pct": (
                (profit_new - profit_base) / abs(profit_base) * 100
                if profit_base != 0 else None
            ),
        }

    return {"error": f"Unknown tool: {name}"}


# ── provider-specific chat implementations ────────────────────────────────────
def _chat_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    import anthropic
    client = anthropic.Anthropic()
    msgs = list(messages)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=_TOOLS_ANTHROPIC,
        messages=msgs,
    )

    while response.stop_reason == "tool_use":
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        for tu in tool_uses:
            result = _run_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })
        msgs.append({"role": "assistant", "content": response.content})
        msgs.append({"role": "user", "content": tool_results})
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=_TOOLS_ANTHROPIC,
            messages=msgs,
        )

    reply = "\n".join(b.text for b in response.content if hasattr(b, "text"))
    msgs.append({"role": "assistant", "content": reply})
    return reply, msgs


def _chat_groq(messages: list[dict]) -> tuple[str, list[dict]]:
    from groq import Groq
    client = Groq()
    # Groq needs system message prepended in the messages list
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=msgs,
        tools=_TOOLS_OPENAI,
        tool_choice="auto",
        max_tokens=1024,
    )

    # agentic loop
    while response.choices[0].finish_reason == "tool_calls":
        msg = response.choices[0].message
        msgs.append(msg)  # assistant message with tool_calls

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _run_tool(tc.function.name, args)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=msgs,
            tools=_TOOLS_OPENAI,
            tool_choice="auto",
            max_tokens=1024,
        )

    reply = response.choices[0].message.content or ""
    # strip system message from returned history
    updated = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "system")]
    updated.append({"role": "assistant", "content": reply})
    return reply, updated


# ── public interface ──────────────────────────────────────────────────────────
def chat(messages: list[dict]) -> tuple[str, list[dict]]:
    """
    Send conversation to the configured LLM with tool use.
    Provider is selected via LLM_PROVIDER / API key env vars.
    """
    if _provider == "groq":
        return _chat_groq(messages)
    return _chat_anthropic(messages)


def active_provider() -> str:
    return _provider
