"""
LLM-powered analytical insight module.

Uses an OpenAI-compatible chat model via LangChain to turn statistical
results into a structured analytical report. Falls back to a rule-based
summary when no API key is configured.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from src.config import load_settings


@dataclass
class InsightReport:
    """Structured analytical report produced by the AI module."""

    summary: str
    key_findings: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    source: str = "rule_based"  # rule_based | llm
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        findings = "\n".join(f"- {f}" for f in self.key_findings)
        drivers = "\n".join(f"- {d}" for d in self.drivers)
        recs = "\n".join(f"- {r}" for r in self.recommendations)
        return (
            f"### Summary\n{self.summary}\n\n"
            f"### Key Findings\n{findings}\n\n"
            f"### Potential Drivers\n{drivers}\n\n"
            f"### Recommendations\n{recs}\n"
        )


SYSTEM_PROMPT = """You are a senior digital economy analyst.
Given statistical analysis results (descriptive stats, correlations,
regression, trends, and optional forecasts), produce a concise executive
analytical report as valid JSON with exactly these keys:

{
  "summary": "2-4 sentence executive summary",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "drivers": ["driver 1", "driver 2", "driver 3"],
  "recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
}

Rules:
- Be specific and quantitative when numbers are available.
- Do not invent data that is not in the context.
- Keep each bullet under 40 words.
- Output JSON only, no markdown fences.
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, tolerating markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # Fallback: find first {...} block
    if not text.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)
    return json.loads(text)


def rule_based_insights(
    analytics_context: dict[str, Any],
    user_question: str = "Analyze the digital industry trend",
) -> InsightReport:
    """
    Deterministic fallback insights derived from analytics artifacts.
    Useful for offline demos and CI without API keys.
    """
    regression = analytics_context.get("regression", {})
    correlations = analytics_context.get("correlations", {}).get("vs_revenue_growth", [])
    trends = analytics_context.get("trend_summary", {}).get("by_industry_latest", [])
    forecast = analytics_context.get("forecast", {})

    r2 = regression.get("r_squared")
    significant = regression.get("significant_features") or []
    coefs = regression.get("coefficients") or {}

    top_corr = correlations[0] if correlations else None
    top_industry = None
    if trends:
        top_industry = max(trends, key=lambda r: r.get("digital_index", 0))

    summary_parts = [
        f"Analysis for: {user_question}.",
        "The platform evaluated digital economy, technology investment, "
        "R&D intensity, employment, and revenue growth indicators.",
    ]
    if r2 is not None:
        summary_parts.append(
            f"A multivariate linear model explains about {r2:.1%} of variation "
            f"in {regression.get('target', 'revenue_growth')}."
        )
    if forecast:
        summary_parts.append(
            f"Forecast method={forecast.get('method')} projects continued "
            f"movement in {forecast.get('metric', 'the focal indicator')}."
        )

    key_findings: list[str] = []
    if top_corr:
        key_findings.append(
            f"{top_corr['feature']} shows the strongest association with revenue growth "
            f"(r={top_corr['correlation']}, p={top_corr['p_value']})."
        )
    if significant:
        ranked = sorted(significant, key=lambda f: abs(coefs.get(f, 0)), reverse=True)
        key_findings.append(
            "Statistically significant growth drivers: " + ", ".join(ranked) + "."
        )
    if top_industry:
        key_findings.append(
            f"{top_industry.get('industry')} leads on digital index "
            f"({top_industry.get('digital_index')}), with annual slope "
            f"{top_industry.get('annual_slope')}."
        )
    if not key_findings:
        key_findings.append("Digital indicators exhibit multi-year upward trends across regions.")

    drivers: list[str] = []
    for feat in (significant or list(coefs.keys()))[:4]:
        direction = "higher" if coefs.get(feat, 0) >= 0 else "lower"
        drivers.append(
            f"{feat.replace('_', ' ').title()} is linked to {direction} revenue growth "
            f"(standardized coef={coefs.get(feat)})."
        )
    if top_corr and top_corr["feature"] not in (significant or []):
        drivers.append(
            f"Correlation evidence highlights {top_corr['feature']} as a key co-movement factor."
        )
    if not drivers:
        drivers = [
            "Technology investment intensity",
            "Digital adoption (digital index)",
            "R&D spending ratios",
        ]

    recommendations = [
        "Prioritize technology investment in industries where digital development index lags peers.",
        "Couple R&D intensity gains with workforce upskilling to convert digital adoption into revenue growth.",
        "Use regional benchmarking dashboards to target capacity-building where CAGR gaps are largest.",
        "Re-run forecasting quarterly and refresh AI summaries after each data refresh cycle.",
    ]

    return InsightReport(
        summary=" ".join(summary_parts),
        key_findings=key_findings[:5],
        drivers=drivers[:5],
        recommendations=recommendations[:4],
        source="rule_based",
    )


def _build_langchain_chain(settings: dict[str, Any]):
    """Construct a LangChain chat chain for structured insight generation."""
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain dependencies missing. Install: pip install langchain langchain-openai"
        ) from exc

    llm_cfg = settings.get("llm", {})
    api_key = llm_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not configured")

    kwargs: dict[str, Any] = {
        "model": llm_cfg.get("model", "gpt-4o-mini"),
        "temperature": float(llm_cfg.get("temperature", 0.3)),
        "max_tokens": int(llm_cfg.get("max_tokens", 1500)),
        "api_key": api_key,
    }
    base_url = llm_cfg.get("base_url") or os.getenv("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url

    llm = ChatOpenAI(**kwargs)

    def invoke(prompt: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return str(response.content)

    return invoke


def generate_insights(
    analytics_context: dict[str, Any],
    user_question: str = "Analyze the digital industry trend",
    use_llm: bool = True,
) -> InsightReport:
    """
    Generate a structured analytical report.

    Args:
        analytics_context: Dict from AnalyticsBundle / forecast artifacts.
        user_question: Natural-language analysis prompt from the user.
        use_llm: If False, skip LLM and use rule-based fallback.
    """
    settings = load_settings()
    context_json = json.dumps(analytics_context, default=str)[:12000]
    user_prompt = (
        f"User question: {user_question}\n\n"
        f"Statistical context (JSON):\n{context_json}"
    )

    if not use_llm:
        return rule_based_insights(analytics_context, user_question)

    api_key = settings.get("llm", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        report = rule_based_insights(analytics_context, user_question)
        report.raw_response = "No API key; returned rule-based insights."
        return report

    try:
        invoke = _build_langchain_chain(settings)
        raw = invoke(user_prompt)
        parsed = _extract_json(raw)
        return InsightReport(
            summary=str(parsed.get("summary", "")).strip(),
            key_findings=[str(x) for x in parsed.get("key_findings", [])],
            drivers=[str(x) for x in parsed.get("drivers", [])],
            recommendations=[str(x) for x in parsed.get("recommendations", [])],
            source="llm",
            raw_response=raw,
        )
    except Exception as exc:  # noqa: BLE001
        report = rule_based_insights(analytics_context, user_question)
        report.raw_response = f"LLM error ({exc}); returned rule-based insights."
        return report


def build_context_from_dataframe(
    df,
    include_forecast: bool = True,
) -> dict[str, Any]:
    """Convenience helper: run analytics (+ optional forecast) → context dict."""
    from src.analytics import run_full_analytics
    from src.forecasting import run_forecast

    bundle = run_full_analytics(df).to_dict()
    if include_forecast:
        try:
            fc = run_forecast(df)
            bundle["forecast"] = {
                "metric": fc.metric,
                "method": fc.method,
                "metrics": fc.metrics,
                "forecast": fc.forecast.to_dict(orient="records"),
            }
        except Exception as exc:  # noqa: BLE001
            bundle["forecast"] = {"error": str(exc)}
    return bundle
