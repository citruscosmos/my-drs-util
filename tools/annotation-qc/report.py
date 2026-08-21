#!/usr/bin/env python3
"""LLM応答のパース(フォールバック込み)とフレーム単位の結果集計。"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from llm_client import ChatClient
from qc_schema import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_fences(text: str) -> str:
    text = _THINK_RE.sub("", text)
    return _FENCE_RE.sub("", text).strip()


def _extract_first_json_object(text: str) -> str | None:
    """テキスト中の最初のバランスの取れた {...} を抜き出す。

    「<think>...</think>」ブロックや「Here's the JSON:」のような前置き、
    末尾の余計なコメントが混ざっていても本体のJSONオブジェクトだけを拾えるようにする
    (thinking対応モデルが推論過程を吐いてJSON全体が単純なjson.loadsで通らないケースへの保険)。
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_qc_response(text: str) -> tuple[dict | None, bool]:
    """(parsed_dict_or_None, parse_ok) を返す。"""
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        pass
    candidate = _extract_first_json_object(cleaned)
    if candidate is not None:
        try:
            return json.loads(candidate), True
        except json.JSONDecodeError:
            pass
    return None, False


def query_with_retry(
    client: ChatClient, user_prompt: str, image_paths: list[str], max_tokens: int = 2048,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> tuple[dict, bool, str]:
    """LLMに問い合わせ、JSONが壊れていたら1回だけリトライ。それでも失敗したら合成フォールバック行を返す。

    system_promptを差し替えれば、annotation QC以外の用途(例: 拡張・合成データの自然さチェック)
    にもこの共通のリトライ/フォールバック機構をそのまま使える。

    戻り値: (parsed_dict, parse_ok, raw_text_of_last_response)
    """
    raw = client.chat_vision(system_prompt, user_prompt, image_paths, max_tokens=max_tokens)
    parsed, ok = parse_qc_response(raw)
    if ok:
        return parsed, True, raw

    retry_prompt = user_prompt + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
    raw2 = client.chat_vision(system_prompt, retry_prompt, image_paths, max_tokens=max_tokens)
    parsed2, ok2 = parse_qc_response(raw2)
    if ok2:
        return parsed2, True, raw2

    fallback = {
        "frame_ok": None,
        "issues": [{
            "category": "other", "severity": "high",
            "description": "LLM output unparsable as JSON after retry", "region": None, "landmark": None, "instance_name": None,
        }],
        "overall_notes": "",
        "raw_response": raw2,
    }
    return fallback, False, raw2


class ReportWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "report.jsonl"
        self.rows: list[dict] = []
        self._fh = open(self.jsonl_path, "w")

    def add(self, clip_id: str, frame_index: int, key: str, channel: str, viz_png_path: str,
             llm_model: str | None, frame_ok, issues: list[dict], overall_notes: str, parse_ok: bool):
        row = {
            "clip_id": clip_id, "frame_index": frame_index, "key": key, "channel": channel,
            "viz_png_path": viz_png_path, "llm_model": llm_model, "frame_ok": frame_ok,
            "issues": issues, "overall_notes": overall_notes, "parse_ok": parse_ok,
        }
        self.rows.append(row)
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()

    def finalize(self):
        self._fh.close()
        if self.rows:
            df = pd.DataFrame(self.rows)
            df.to_parquet(self.out_dir / "report.parquet", engine="pyarrow", index=False)

        cat_counter = Counter()
        sev_counter = Counter()
        n_ok, n_evaluated, n_total = 0, 0, 0
        for row in self.rows:
            n_total += 1
            # frame_ok is None when --skip-llm was used (not evaluated, not "failed")
            if row["frame_ok"] is not None:
                n_evaluated += 1
                if row["frame_ok"]:
                    n_ok += 1
            for issue in row["issues"]:
                cat_counter[issue.get("category", "other")] += 1
                sev_counter[issue.get("severity", "low")] += 1

        summary = {
            "n_frames_checked": n_total,
            "n_frames_evaluated_by_llm": n_evaluated,
            "frame_ok_ratio": (n_ok / n_evaluated) if n_evaluated else None,
            "issue_counts_by_category": dict(cat_counter),
            "issue_counts_by_severity": dict(sev_counter),
        }
        (self.out_dir / "report_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return summary
