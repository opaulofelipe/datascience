#!/usr/bin/env python3
"""
DataAudit Pro
=============

Ferramenta de auditoria, limpeza, perfilamento e conciliacao de dados.

Casos de uso:
- comparar relatorios de sistemas diferentes;
- encontrar transacoes ausentes;
- detectar valores divergentes;
- localizar duplicidades;
- validar campos obrigatorios;
- detectar categorias invalidas;
- identificar valores extremos;
- padronizar CSV/Excel;
- gerar relatorios automaticamente.

Dependencias:
    pip install pandas numpy openpyxl

Exemplos:
    python data_audit_pro.py sample --output demo

    python data_audit_pro.py audit \
        --input demo/vendas_sistema.csv \
        --config demo/config.json \
        --output saida

    python data_audit_pro.py reconcile \
        --left demo/vendas_sistema.csv \
        --right demo/vendas_financeiro.xlsx \
        --config demo/config.json \
        --output saida
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import logging
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger("data_audit_pro")


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# MODELOS DE DADOS
# =============================================================================

@dataclass
class Issue:
    dataset: str
    rule: str
    severity: str
    row_number: Optional[int]
    column: Optional[str]
    value: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "rule": self.rule,
            "severity": self.severity,
            "row_number": self.row_number,
            "column": self.column,
            "value": json_safe(self.value),
            "message": self.message,
        }


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    rows: int
    non_null: int
    nulls: int
    null_pct: float
    unique: int
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    min_value: Any = None
    max_value: Any = None
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    outliers_iqr: Optional[int] = None


@dataclass
class DatasetProfile:
    name: str
    rows: int
    columns: int
    duplicate_rows: int
    memory_mb: float
    file_hash_sha256: Optional[str]
    column_profiles: list[ColumnProfile] = field(default_factory=list)


@dataclass
class ReconciliationSummary:
    left_rows: int
    right_rows: int
    matched_rows: int
    left_only_rows: int
    right_only_rows: int
    matched_with_differences: int
    perfect_matches: int
    match_rate_left_pct: float
    match_rate_right_pct: float


# =============================================================================
# CONFIGURACAO
# =============================================================================

DEFAULT_CONFIG = {
    "column_aliases": {
        "id": [
            "id",
            "codigo",
            "código",
            "id_transacao",
            "transacao_id",
            "ID Transação",
        ],
        "data": [
            "data",
            "dt",
            "data_venda",
            "data_transacao",
            "Data Venda",
        ],
        "cliente": [
            "cliente",
            "nome_cliente",
            "customer",
            "Nome Cliente",
        ],
        "categoria": [
            "categoria",
            "tipo",
            "grupo",
        ],
        "valor": [
            "valor",
            "vlr",
            "total",
            "valor_total",
            "amount",
            "Valor Total",
        ],
        "status": [
            "status",
            "situacao",
            "situação",
            "Situação",
        ],
    },
    "cleaning": {
        "strip_strings": True,
        "collapse_spaces": True,
        "lowercase_columns": [
            "status",
            "categoria",
        ],
        "uppercase_columns": [],
        "date_columns": [
            "data",
        ],
        "numeric_columns": [
            "valor",
        ],
        "decimal_comma": True,
    },
    "validation_rules": [
        {
            "type": "required",
            "columns": [
                "id",
                "data",
                "cliente",
                "valor",
                "status",
            ],
            "severity": "error",
        },
        {
            "type": "unique",
            "columns": [
                "id",
            ],
            "severity": "error",
        },
        {
            "type": "range",
            "column": "valor",
            "min": 0,
            "max": 1000000,
            "severity": "error",
        },
        {
            "type": "allowed_values",
            "column": "status",
            "values": [
                "pago",
                "pendente",
                "cancelado",
            ],
            "severity": "warning",
        },
        {
            "type": "regex",
            "column": "id",
            "pattern": "^V[0-9]{5}$",
            "severity": "warning",
        },
    ],
    "reconciliation": {
        "keys": [
            "id",
        ],
        "compare_columns": [
            "data",
            "cliente",
            "categoria",
            "valor",
            "status",
        ],
        "numeric_tolerance": {
            "valor": 0.01,
        },
        "ignore_case_columns": [
            "cliente",
            "categoria",
            "status",
        ],
        "ignore_whitespace_columns": [
            "cliente",
            "categoria",
            "status",
        ],
    },
}


class ConfigManager:
    @staticmethod
    def load(path: Optional[str]) -> dict[str, Any]:
        if not path:
            LOGGER.info(
                "Configuracao nao informada. Usando configuracao padrao."
            )
            return json.loads(json.dumps(DEFAULT_CONFIG))

        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuracao nao encontrado: {config_path}"
            )

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            user_config = json.load(file)

        return ConfigManager.deep_merge(
            json.loads(json.dumps(DEFAULT_CONFIG)),
            user_config,
        )

    @staticmethod
    def deep_merge(
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        for key, value in override.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                base[key] = ConfigManager.deep_merge(
                    base[key],
                    value,
                )
            else:
                base[key] = value

        return base

    @staticmethod
    def save_default(path: Path) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            json.dumps(
                DEFAULT_CONFIG,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


# =============================================================================
# UTILITARIOS
# =============================================================================

def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = unicodedata.normalize(
        "NFKD",
        text,
    )
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = text.lower()
    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )
    text = re.sub(
        r"_+",
        "_",
        text,
    )
    return text.strip("_")


def file_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        if np.isnan(value):
            return None
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        float,
    ) and np.isnan(value):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value


def ensure_output_dir(
    path: str | Path,
) -> Path:
    output = Path(path)
    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output


# =============================================================================
# LEITURA
# =============================================================================

class DataLoader:
    SUPPORTED = {
        ".csv",
        ".xlsx",
        ".xls",
    }

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> pd.DataFrame:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Arquivo nao encontrado: {file_path}"
            )

        suffix = file_path.suffix.lower()

        if suffix not in cls.SUPPORTED:
            supported = ", ".join(
                sorted(cls.SUPPORTED)
            )
            raise ValueError(
                f"Formato {suffix} nao suportado. "
                f"Formatos aceitos: {supported}"
            )

        LOGGER.info(
            "Lendo %s",
            file_path,
        )

        if suffix == ".csv":
            return cls._read_csv(file_path)

        return pd.read_excel(file_path)

    @staticmethod
    def _read_csv(
        path: Path,
    ) -> pd.DataFrame:
        encodings = [
            "utf-8-sig",
            "utf-8",
            "latin1",
            "cp1252",
        ]

        last_error: Optional[Exception] = None

        for encoding in encodings:
            try:
                sample = path.read_text(
                    encoding=encoding,
                )[:5000]

                try:
                    dialect = csv.Sniffer().sniff(
                        sample,
                        delimiters=",;\t|",
                    )
                    separator = dialect.delimiter
                except csv.Error:
                    separator = ","

                LOGGER.debug(
                    "CSV detectado: encoding=%s separador=%r",
                    encoding,
                    separator,
                )

                return pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=separator,
                )

            except Exception as exc:
                last_error = exc

        raise ValueError(
            f"Nao foi possivel ler o CSV: {last_error}"
        )


# =============================================================================
# NORMALIZACAO DE COLUNAS
# =============================================================================

class ColumnNormalizer:
    def __init__(
        self,
        aliases: dict[str, list[str]],
    ):
        self.aliases = aliases
        self.lookup = self._build_lookup(
            aliases
        )

    @staticmethod
    def _build_lookup(
        aliases: dict[str, list[str]],
    ) -> dict[str, str]:
        lookup: dict[str, str] = {}

        for canonical, alternatives in aliases.items():
            lookup[
                normalize_text(canonical)
            ] = canonical

            for alias in alternatives:
                lookup[
                    normalize_text(alias)
                ] = canonical

        return lookup

    def normalize(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        result = df.copy()
        rename_map: dict[str, str] = {}
        used_names: dict[str, int] = {}

        for original in result.columns:
            normalized = normalize_text(
                original
            )

            canonical = self.lookup.get(
                normalized,
                normalized,
            )

            if canonical in used_names:
                used_names[
                    canonical
                ] += 1

                canonical = (
                    f"{canonical}_"
                    f"{used_names[canonical]}"
                )
            else:
                used_names[
                    canonical
                ] = 1

            rename_map[
                original
            ] = canonical

        LOGGER.debug(
            "Colunas normalizadas: %s",
            rename_map,
        )

        return result.rename(
            columns=rename_map
        )


# =============================================================================
# LIMPEZA
# =============================================================================

class DataCleaner:
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    def clean(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        result = df.copy()

        if self.config.get(
            "strip_strings",
            True,
        ):
            result = self._strip_strings(
                result
            )

        if self.config.get(
            "collapse_spaces",
            True,
        ):
            result = self._collapse_spaces(
                result
            )

        for column in self.config.get(
            "lowercase_columns",
            [],
        ):
            if column in result.columns:
                result[
                    column
                ] = result[
                    column
                ].map(
                    lambda value: (
                        value.lower()
                        if isinstance(
                            value,
                            str,
                        )
                        else value
                    )
                )

        for column in self.config.get(
            "uppercase_columns",
            [],
        ):
            if column in result.columns:
                result[
                    column
                ] = result[
                    column
                ].map(
                    lambda value: (
                        value.upper()
                        if isinstance(
                            value,
                            str,
                        )
                        else value
                    )
                )

        for column in self.config.get(
            "date_columns",
            [],
        ):
            if column in result.columns:
                result[
                    column
                ] = self._parse_dates(
                    result[
                        column
                    ]
                )

        for column in self.config.get(
            "numeric_columns",
            [],
        ):
            if column in result.columns:
                result[
                    column
                ] = self._parse_numeric(
                    result[
                        column
                    ],
                    decimal_comma=self.config.get(
                        "decimal_comma",
                        True,
                    ),
                )

        return result

    @staticmethod
    def _strip_strings(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        for column in df.select_dtypes(
            include=[
                "object",
                "string",
            ]
        ).columns:
            df[
                column
            ] = df[
                column
            ].map(
                lambda value: (
                    value.strip()
                    if isinstance(
                        value,
                        str,
                    )
                    else value
                )
            )

        return df

    @staticmethod
    def _collapse_spaces(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        for column in df.select_dtypes(
            include=[
                "object",
                "string",
            ]
        ).columns:
            df[
                column
            ] = df[
                column
            ].map(
                lambda value: (
                    re.sub(
                        r"\s+",
                        " ",
                        value,
                    )
                    if isinstance(
                        value,
                        str,
                    )
                    else value
                )
            )

        return df

    @staticmethod
    def _parse_dates(
        series: pd.Series,
    ) -> pd.Series:
        # Preserva datas que ja chegaram tipadas pelo Excel.
        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            return pd.to_datetime(
                series,
                errors="coerce",
            )

        # Evita a ambiguidade de dayfirst=True em datas ISO como 2026-02-01.
        text = (
            series
            .astype("string")
            .str
            .strip()
        )

        iso_mask = text.str.match(
            r"^\d{4}-\d{1,2}-\d{1,2}(?:[ T].*)?$",
            na=False,
        )

        parsed = pd.Series(
            pd.NaT,
            index=series.index,
            dtype="datetime64[ns]",
        )

        if iso_mask.any():
            parsed.loc[
                iso_mask
            ] = pd.to_datetime(
                text.loc[
                    iso_mask
                ],
                errors="coerce",
                yearfirst=True,
            )

        non_iso_mask = ~iso_mask

        if non_iso_mask.any():
            parsed.loc[
                non_iso_mask
            ] = pd.to_datetime(
                text.loc[
                    non_iso_mask
                ],
                errors="coerce",
                dayfirst=True,
            )

        return parsed

    @staticmethod
    def _parse_numeric(
        series: pd.Series,
        decimal_comma: bool,
    ) -> pd.Series:
        if pd.api.types.is_numeric_dtype(
            series
        ):
            return pd.to_numeric(
                series,
                errors="coerce",
            )

        def convert(
            value: Any,
        ) -> Any:
            if pd.isna(value):
                return np.nan

            text = str(
                value
            ).strip()

            text = re.sub(
                r"[Rr]\$|\s",
                "",
                text,
            )

            if decimal_comma:
                if "," in text:
                    text = (
                        text
                        .replace(
                            ".",
                            "",
                        )
                        .replace(
                            ",",
                            ".",
                        )
                    )
            else:
                text = text.replace(
                    ",",
                    "",
                )

            return text

        return pd.to_numeric(
            series.map(
                convert
            ),
            errors="coerce",
        )


# =============================================================================
# VALIDACAO
# =============================================================================

class DataValidator:
    def __init__(
        self,
        rules: list[dict[str, Any]],
    ):
        self.rules = rules

    def validate(
        self,
        df: pd.DataFrame,
        dataset_name: str,
    ) -> list[Issue]:
        issues: list[Issue] = []

        for rule in self.rules:
            rule_type = rule.get(
                "type"
            )

            severity = rule.get(
                "severity",
                "warning",
            )

            if rule_type == "required":
                issues.extend(
                    self._required(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            elif rule_type == "unique":
                issues.extend(
                    self._unique(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            elif rule_type == "range":
                issues.extend(
                    self._range(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            elif rule_type == "allowed_values":
                issues.extend(
                    self._allowed_values(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            elif rule_type == "regex":
                issues.extend(
                    self._regex(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            elif rule_type == "date_range":
                issues.extend(
                    self._date_range(
                        df,
                        dataset_name,
                        rule,
                        severity,
                    )
                )

            else:
                LOGGER.warning(
                    "Regra desconhecida: %s",
                    rule_type,
                )

        return issues

    @staticmethod
    def _required(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        issues: list[Issue] = []

        for column in rule.get(
            "columns",
            [],
        ):
            if column not in df.columns:
                issues.append(
                    Issue(
                        dataset=dataset_name,
                        rule="required",
                        severity="error",
                        row_number=None,
                        column=column,
                        value=None,
                        message=(
                            "Coluna obrigatoria "
                            f"ausente: {column}"
                        ),
                    )
                )
                continue

            series = df[
                column
            ]

            mask = series.isna()

            if (
                pd.api.types.is_object_dtype(
                    series
                )
                or pd.api.types.is_string_dtype(
                    series
                )
            ):
                empty_strings = (
                    series
                    .astype(
                        "string"
                    )
                    .str
                    .strip()
                    .eq("")
                    .fillna(
                        False
                    )
                )

                mask = (
                    mask
                    | empty_strings
                )

            for index in df.index[
                mask
            ]:
                issues.append(
                    Issue(
                        dataset=dataset_name,
                        rule="required",
                        severity=severity,
                        row_number=int(
                            index
                        ) + 2,
                        column=column,
                        value=df.at[
                            index,
                            column,
                        ],
                        message=(
                            "Campo obrigatorio "
                            f"vazio: {column}"
                        ),
                    )
                )

        return issues

    @staticmethod
    def _unique(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        columns = [
            column
            for column in rule.get(
                "columns",
                [],
            )
            if column in df.columns
        ]

        if not columns:
            return []

        duplicated = df.duplicated(
            subset=columns,
            keep=False,
        )

        issues: list[Issue] = []

        for index in df.index[
            duplicated
        ]:
            value = {
                column: json_safe(
                    df.at[
                        index,
                        column,
                    ]
                )
                for column in columns
            }

            issues.append(
                Issue(
                    dataset=dataset_name,
                    rule="unique",
                    severity=severity,
                    row_number=int(
                        index
                    ) + 2,
                    column=", ".join(
                        columns
                    ),
                    value=value,
                    message=(
                        "Chave duplicada "
                        f"em {columns}."
                    ),
                )
            )

        return issues

    @staticmethod
    def _range(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        column = rule[
            "column"
        ]

        if column not in df.columns:
            return []

        values = pd.to_numeric(
            df[
                column
            ],
            errors="coerce",
        )

        minimum = rule.get(
            "min"
        )

        maximum = rule.get(
            "max"
        )

        mask = pd.Series(
            False,
            index=df.index,
        )

        if minimum is not None:
            mask = (
                mask
                | (
                    values
                    < minimum
                )
            )

        if maximum is not None:
            mask = (
                mask
                | (
                    values
                    > maximum
                )
            )

        issues: list[Issue] = []

        for index in df.index[
            mask.fillna(
                False
            )
        ]:
            issues.append(
                Issue(
                    dataset=dataset_name,
                    rule="range",
                    severity=severity,
                    row_number=int(
                        index
                    ) + 2,
                    column=column,
                    value=df.at[
                        index,
                        column,
                    ],
                    message=(
                        f"Valor fora da faixa "
                        f"[{minimum}, {maximum}]."
                    ),
                )
            )

        return issues

    @staticmethod
    def _allowed_values(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        column = rule[
            "column"
        ]

        if column not in df.columns:
            return []

        allowed = set(
            rule.get(
                "values",
                [],
            )
        )

        mask = (
            df[
                column
            ].notna()
            & ~df[
                column
            ].isin(
                allowed
            )
        )

        issues: list[Issue] = []

        for index in df.index[
            mask
        ]:
            issues.append(
                Issue(
                    dataset=dataset_name,
                    rule="allowed_values",
                    severity=severity,
                    row_number=int(
                        index
                    ) + 2,
                    column=column,
                    value=df.at[
                        index,
                        column,
                    ],
                    message=(
                        f"Valor nao permitido. "
                        f"Aceitos: {sorted(allowed)}"
                    ),
                )
            )

        return issues

    @staticmethod
    def _regex(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        column = rule[
            "column"
        ]

        if column not in df.columns:
            return []

        pattern = re.compile(
            rule[
                "pattern"
            ]
        )

        mask = (
            df[
                column
            ].notna()
            & ~df[
                column
            ].astype(
                str
            ).map(
                lambda value: bool(
                    pattern.fullmatch(
                        value
                    )
                )
            )
        )

        issues: list[Issue] = []

        for index in df.index[
            mask
        ]:
            issues.append(
                Issue(
                    dataset=dataset_name,
                    rule="regex",
                    severity=severity,
                    row_number=int(
                        index
                    ) + 2,
                    column=column,
                    value=df.at[
                        index,
                        column,
                    ],
                    message=(
                        "Valor fora do padrao "
                        f"{rule['pattern']}."
                    ),
                )
            )

        return issues

    @staticmethod
    def _date_range(
        df: pd.DataFrame,
        dataset_name: str,
        rule: dict[str, Any],
        severity: str,
    ) -> list[Issue]:
        column = rule[
            "column"
        ]

        if column not in df.columns:
            return []

        values = pd.to_datetime(
            df[
                column
            ],
            errors="coerce",
        )

        start = (
            pd.to_datetime(
                rule[
                    "start"
                ]
            )
            if rule.get(
                "start"
            )
            else None
        )

        end = (
            pd.to_datetime(
                rule[
                    "end"
                ]
            )
            if rule.get(
                "end"
            )
            else None
        )

        mask = pd.Series(
            False,
            index=df.index,
        )

        if start is not None:
            mask = (
                mask
                | (
                    values
                    < start
                )
            )

        if end is not None:
            mask = (
                mask
                | (
                    values
                    > end
                )
            )

        issues: list[Issue] = []

        for index in df.index[
            mask.fillna(
                False
            )
        ]:
            issues.append(
                Issue(
                    dataset=dataset_name,
                    rule="date_range",
                    severity=severity,
                    row_number=int(
                        index
                    ) + 2,
                    column=column,
                    value=df.at[
                        index,
                        column,
                    ],
                    message=(
                        "Data fora do intervalo "
                        "configurado."
                    ),
                )
            )

        return issues


# =============================================================================
# PERFILAMENTO
# =============================================================================

class DataProfiler:
    @staticmethod
    def profile(
        df: pd.DataFrame,
        dataset_name: str,
        file_hash: Optional[str] = None,
    ) -> DatasetProfile:
        row_count = len(
            df
        )

        profiles: list[
            ColumnProfile
        ] = []

        for column in df.columns:
            series = df[
                column
            ]

            non_null = int(
                series.notna().sum()
            )

            nulls = int(
                series.isna().sum()
            )

            unique = int(
                series.nunique(
                    dropna=True
                )
            )

            profile = ColumnProfile(
                name=column,
                dtype=str(
                    series.dtype
                ),
                rows=row_count,
                non_null=non_null,
                nulls=nulls,
                null_pct=round(
                    (
                        nulls
                        / row_count
                        * 100
                    )
                    if row_count
                    else 0,
                    2,
                ),
                unique=unique,
                unique_pct=round(
                    (
                        unique
                        / row_count
                        * 100
                    )
                    if row_count
                    else 0,
                    2,
                ),
                sample_values=[
                    json_safe(
                        value
                    )
                    for value in (
                        series
                        .dropna()
                        .drop_duplicates()
                        .head(5)
                        .tolist()
                    )
                ],
            )

            if pd.api.types.is_numeric_dtype(
                series
            ):
                numeric = pd.to_numeric(
                    series,
                    errors="coerce",
                ).dropna()

                if not numeric.empty:
                    profile.min_value = json_safe(
                        numeric.min()
                    )

                    profile.max_value = json_safe(
                        numeric.max()
                    )

                    profile.mean = round(
                        float(
                            numeric.mean()
                        ),
                        4,
                    )

                    profile.median = round(
                        float(
                            numeric.median()
                        ),
                        4,
                    )

                    profile.std = round(
                        float(
                            numeric.std()
                        ),
                        4,
                    ) if len(
                        numeric
                    ) > 1 else 0.0

                    profile.outliers_iqr = (
                        DataProfiler
                        .count_iqr_outliers(
                            numeric
                        )
                    )

            elif pd.api.types.is_datetime64_any_dtype(
                series
            ):
                dates = series.dropna()

                if not dates.empty:
                    profile.min_value = json_safe(
                        dates.min()
                    )

                    profile.max_value = json_safe(
                        dates.max()
                    )

            profiles.append(
                profile
            )

        return DatasetProfile(
            name=dataset_name,
            rows=row_count,
            columns=len(
                df.columns
            ),
            duplicate_rows=int(
                df.duplicated().sum()
            ),
            memory_mb=round(
                float(
                    df.memory_usage(
                        deep=True
                    ).sum()
                    / (
                        1024 ** 2
                    )
                ),
                4,
            ),
            file_hash_sha256=file_hash,
            column_profiles=profiles,
        )

    @staticmethod
    def count_iqr_outliers(
        series: pd.Series,
    ) -> int:
        if len(
            series
        ) < 4:
            return 0

        q1 = series.quantile(
            0.25
        )

        q3 = series.quantile(
            0.75
        )

        iqr = q3 - q1

        if iqr == 0:
            return 0

        lower = (
            q1
            - 1.5
            * iqr
        )

        upper = (
            q3
            + 1.5
            * iqr
        )

        return int(
            (
                (
                    series
                    < lower
                )
                | (
                    series
                    > upper
                )
            ).sum()
        )


# =============================================================================
# CONCILIACAO
# =============================================================================

class DataReconciler:
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

    def reconcile(
        self,
        left: pd.DataFrame,
        right: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        ReconciliationSummary,
    ]:
        keys = self.config.get(
            "keys",
            [],
        )

        compare_columns = self.config.get(
            "compare_columns",
            [],
        )

        tolerances = self.config.get(
            "numeric_tolerance",
            {},
        )

        ignore_case = set(
            self.config.get(
                "ignore_case_columns",
                [],
            )
        )

        ignore_whitespace = set(
            self.config.get(
                "ignore_whitespace_columns",
                [],
            )
        )

        if not keys:
            raise ValueError(
                "Nenhuma chave de conciliacao configurada."
            )

        missing_left = [
            key
            for key in keys
            if key not in left.columns
        ]

        missing_right = [
            key
            for key in keys
            if key not in right.columns
        ]

        if missing_left or missing_right:
            raise ValueError(
                "Chaves ausentes. "
                f"Esquerda={missing_left}; "
                f"Direita={missing_right}"
            )

        if left.duplicated(
            subset=keys
        ).any():
            LOGGER.warning(
                "A base esquerda possui chaves duplicadas."
            )

        if right.duplicated(
            subset=keys
        ).any():
            LOGGER.warning(
                "A base direita possui chaves duplicadas."
            )

        merged = left.merge(
            right,
            how="outer",
            on=keys,
            suffixes=(
                "_left",
                "_right",
            ),
            indicator=True,
        )

        merged[
            "reconciliation_status"
        ] = merged[
            "_merge"
        ].map(
            {
                "left_only": "somente_esquerda",
                "right_only": "somente_direita",
                "both": "encontrado_nas_duas",
            }
        ).astype(
            "object"
        )

        merged[
            "differences"
        ] = ""

        both_mask = merged[
            "_merge"
        ].eq(
            "both"
        )

        for index in merged.index[
            both_mask
        ]:
            differences: list[
                str
            ] = []

            for column in compare_columns:
                left_column = (
                    f"{column}_left"
                )

                right_column = (
                    f"{column}_right"
                )

                if (
                    left_column
                    not in merged.columns
                    or right_column
                    not in merged.columns
                ):
                    continue

                left_value = merged.at[
                    index,
                    left_column,
                ]

                right_value = merged.at[
                    index,
                    right_column,
                ]

                is_equal = (
                    self.values_equal(
                        left_value,
                        right_value,
                        tolerance=tolerances.get(
                            column
                        ),
                        ignore_case=(
                            column
                            in ignore_case
                        ),
                        ignore_whitespace=(
                            column
                            in ignore_whitespace
                        ),
                    )
                )

                if not is_equal:
                    differences.append(
                        column
                    )

            merged.at[
                index,
                "differences",
            ] = ", ".join(
                differences
            )

            if differences:
                merged.at[
                    index,
                    "reconciliation_status",
                ] = "divergente"
            else:
                merged.at[
                    index,
                    "reconciliation_status",
                ] = "conciliado"

        left_only = int(
            (
                merged[
                    "_merge"
                ]
                == "left_only"
            ).sum()
        )

        right_only = int(
            (
                merged[
                    "_merge"
                ]
                == "right_only"
            ).sum()
        )

        divergent = int(
            (
                merged[
                    "reconciliation_status"
                ]
                == "divergente"
            ).sum()
        )

        perfect = int(
            (
                merged[
                    "reconciliation_status"
                ]
                == "conciliado"
            ).sum()
        )

        matched = (
            divergent
            + perfect
        )

        summary = ReconciliationSummary(
            left_rows=len(
                left
            ),
            right_rows=len(
                right
            ),
            matched_rows=matched,
            left_only_rows=left_only,
            right_only_rows=right_only,
            matched_with_differences=divergent,
            perfect_matches=perfect,
            match_rate_left_pct=round(
                (
                    matched
                    / len(left)
                    * 100
                )
                if len(
                    left
                )
                else 0,
                2,
            ),
            match_rate_right_pct=round(
                (
                    matched
                    / len(right)
                    * 100
                )
                if len(
                    right
                )
                else 0,
                2,
            ),
        )

        result = merged.drop(
            columns=[
                "_merge",
            ]
        )

        return (
            result,
            summary,
        )

    @staticmethod
    def values_equal(
        left: Any,
        right: Any,
        tolerance: Optional[float] = None,
        ignore_case: bool = False,
        ignore_whitespace: bool = False,
    ) -> bool:
        left_missing = pd.isna(
            left
        )

        right_missing = pd.isna(
            right
        )

        if (
            left_missing
            and right_missing
        ):
            return True

        if (
            left_missing
            != right_missing
        ):
            return False

        if tolerance is not None:
            try:
                return (
                    abs(
                        float(left)
                        - float(right)
                    )
                    <= float(
                        tolerance
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

        if (
            isinstance(
                left,
                (
                    pd.Timestamp,
                    datetime,
                ),
            )
            or isinstance(
                right,
                (
                    pd.Timestamp,
                    datetime,
                ),
            )
        ):
            try:
                return (
                    pd.Timestamp(
                        left
                    )
                    == pd.Timestamp(
                        right
                    )
                )
            except Exception:
                return False

        left_text = str(
            left
        )

        right_text = str(
            right
        )

        if ignore_whitespace:
            left_text = re.sub(
                r"\s+",
                " ",
                left_text,
            ).strip()

            right_text = re.sub(
                r"\s+",
                " ",
                right_text,
            ).strip()

        if ignore_case:
            left_text = (
                left_text
                .casefold()
            )

            right_text = (
                right_text
                .casefold()
            )

        return (
            left_text
            == right_text
        )


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

class AuditPipeline:
    def __init__(
        self,
        config: dict[str, Any],
    ):
        self.config = config

        self.normalizer = ColumnNormalizer(
            config.get(
                "column_aliases",
                {},
            )
        )

        self.cleaner = DataCleaner(
            config.get(
                "cleaning",
                {},
            )
        )

        self.validator = DataValidator(
            config.get(
                "validation_rules",
                [],
            )
        )

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        normalized = (
            self.normalizer
            .normalize(
                df
            )
        )

        cleaned = (
            self.cleaner
            .clean(
                normalized
            )
        )

        return cleaned

    def audit_file(
        self,
        path: str | Path,
        dataset_name: Optional[str] = None,
    ) -> tuple[
        pd.DataFrame,
        DatasetProfile,
        list[Issue],
    ]:
        file_path = Path(
            path
        )

        name = (
            dataset_name
            or file_path.stem
        )

        raw = DataLoader.load(
            file_path
        )

        cleaned = self.prepare(
            raw
        )

        profile = DataProfiler.profile(
            cleaned,
            dataset_name=name,
            file_hash=file_sha256(
                file_path
            ),
        )

        issues = self.validator.validate(
            cleaned,
            name,
        )

        LOGGER.info(
            "%s: %d linhas, %d colunas, %d problema(s)",
            name,
            len(
                cleaned
            ),
            len(
                cleaned.columns
            ),
            len(
                issues
            ),
        )

        return (
            cleaned,
            profile,
            issues,
        )


# =============================================================================
# RELATORIOS
# =============================================================================

class ReportWriter:
    def __init__(
        self,
        output_dir: str | Path,
    ):
        self.output_dir = ensure_output_dir(
            output_dir
        )

    def write_audit(
        self,
        cleaned_df: pd.DataFrame,
        profile: DatasetProfile,
        issues: list[Issue],
        base_name: str,
    ) -> None:
        issues_df = pd.DataFrame(
            [
                issue.to_dict()
                for issue in issues
            ]
        )

        profile_df = pd.DataFrame(
            [
                asdict(
                    column
                )
                for column
                in profile.column_profiles
            ]
        )

        cleaned_path = (
            self.output_dir
            / f"{base_name}_cleaned.csv"
        )

        cleaned_df.to_csv(
            cleaned_path,
            index=False,
            encoding="utf-8-sig",
        )

        excel_path = (
            self.output_dir
            / f"{base_name}_audit.xlsx"
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        ) as writer:
            cleaned_df.to_excel(
                writer,
                sheet_name="dados_limpos",
                index=False,
            )

            if issues_df.empty:
                pd.DataFrame(
                    [
                        {
                            "mensagem": (
                                "Nenhum problema encontrado."
                            )
                        }
                    ]
                ).to_excel(
                    writer,
                    sheet_name="problemas",
                    index=False,
                )
            else:
                issues_df.to_excel(
                    writer,
                    sheet_name="problemas",
                    index=False,
                )

            profile_df.to_excel(
                writer,
                sheet_name="perfil_colunas",
                index=False,
            )

            pd.DataFrame(
                [
                    {
                        "dataset": profile.name,
                        "linhas": profile.rows,
                        "colunas": profile.columns,
                        "linhas_duplicadas": (
                            profile.duplicate_rows
                        ),
                        "memoria_mb": profile.memory_mb,
                        "sha256": (
                            profile.file_hash_sha256
                        ),
                        "total_problemas": len(
                            issues
                        ),
                        "erros": sum(
                            issue.severity
                            == "error"
                            for issue in issues
                        ),
                        "avisos": sum(
                            issue.severity
                            == "warning"
                            for issue in issues
                        ),
                    }
                ]
            ).to_excel(
                writer,
                sheet_name="resumo",
                index=False,
            )

        json_path = (
            self.output_dir
            / f"{base_name}_summary.json"
        )

        json_path.write_text(
            json.dumps(
                {
                    "profile": self.profile_to_json(
                        profile
                    ),
                    "issues": [
                        issue.to_dict()
                        for issue in issues
                    ],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        html_path = (
            self.output_dir
            / f"{base_name}_report.html"
        )

        html_path.write_text(
            self.build_audit_html(
                profile,
                issues,
            ),
            encoding="utf-8",
        )

        LOGGER.info(
            "Auditoria exportada para %s",
            self.output_dir,
        )

    def write_reconciliation(
        self,
        result: pd.DataFrame,
        summary: ReconciliationSummary,
        left_profile: DatasetProfile,
        right_profile: DatasetProfile,
    ) -> None:
        csv_path = (
            self.output_dir
            / "reconciliation_result.csv"
        )

        result.to_csv(
            csv_path,
            index=False,
            encoding="utf-8-sig",
        )

        excel_path = (
            self.output_dir
            / "reconciliation_report.xlsx"
        )

        with pd.ExcelWriter(
            excel_path,
            engine="openpyxl",
        ) as writer:
            pd.DataFrame(
                [
                    asdict(
                        summary
                    )
                ]
            ).to_excel(
                writer,
                sheet_name="resumo",
                index=False,
            )

            result.to_excel(
                writer,
                sheet_name="resultado_completo",
                index=False,
            )

            for status, sheet_name in [
                (
                    "conciliado",
                    "conciliados",
                ),
                (
                    "divergente",
                    "divergentes",
                ),
                (
                    "somente_esquerda",
                    "somente_esquerda",
                ),
                (
                    "somente_direita",
                    "somente_direita",
                ),
            ]:
                subset = result[
                    result[
                        "reconciliation_status"
                    ]
                    == status
                ]

                subset.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

            pd.DataFrame(
                [
                    asdict(
                        column
                    )
                    for column
                    in left_profile.column_profiles
                ]
            ).to_excel(
                writer,
                sheet_name="perfil_esquerda",
                index=False,
            )

            pd.DataFrame(
                [
                    asdict(
                        column
                    )
                    for column
                    in right_profile.column_profiles
                ]
            ).to_excel(
                writer,
                sheet_name="perfil_direita",
                index=False,
            )

        json_path = (
            self.output_dir
            / "reconciliation_summary.json"
        )

        json_path.write_text(
            json.dumps(
                {
                    "summary": asdict(
                        summary
                    ),
                    "left_profile": (
                        self.profile_to_json(
                            left_profile
                        )
                    ),
                    "right_profile": (
                        self.profile_to_json(
                            right_profile
                        )
                    ),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )

        html_path = (
            self.output_dir
            / "reconciliation_report.html"
        )

        html_path.write_text(
            self.build_reconciliation_html(
                summary,
                result,
            ),
            encoding="utf-8",
        )

        LOGGER.info(
            "Conciliacao exportada para %s",
            self.output_dir,
        )

    @staticmethod
    def profile_to_json(
        profile: DatasetProfile,
    ) -> dict[str, Any]:
        return {
            "name": profile.name,
            "rows": profile.rows,
            "columns": profile.columns,
            "duplicate_rows": (
                profile.duplicate_rows
            ),
            "memory_mb": profile.memory_mb,
            "file_hash_sha256": (
                profile.file_hash_sha256
            ),
            "column_profiles": [
                {
                    key: json_safe(
                        value
                    )
                    for key, value
                    in asdict(
                        column
                    ).items()
                }
                for column
                in profile.column_profiles
            ],
        }

    @staticmethod
    def build_audit_html(
        profile: DatasetProfile,
        issues: list[Issue],
    ) -> str:
        errors = sum(
            issue.severity
            == "error"
            for issue in issues
        )

        warnings = sum(
            issue.severity
            == "warning"
            for issue in issues
        )

        profile_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(column.name)}</td>"
                f"<td>{html.escape(column.dtype)}</td>"
                f"<td>{column.nulls}</td>"
                f"<td>{column.null_pct:.2f}%</td>"
                f"<td>{column.unique}</td>"
                f"<td>{column.unique_pct:.2f}%</td>"
                f"<td>{'' if column.outliers_iqr is None else column.outliers_iqr}</td>"
                "</tr>"
            )
            for column
            in profile.column_profiles
        )

        issue_rows = "".join(
            (
                "<tr>"
                f"<td>{html.escape(issue.severity)}</td>"
                f"<td>{html.escape(issue.rule)}</td>"
                f"<td>{'' if issue.row_number is None else issue.row_number}</td>"
                f"<td>{html.escape(issue.column or '')}</td>"
                f"<td>{html.escape(str(json_safe(issue.value)))}</td>"
                f"<td>{html.escape(issue.message)}</td>"
                "</tr>"
            )
            for issue
            in issues[:500]
        )

        if issue_rows:
            issues_section = (
                "<div class='table-wrap'>"
                "<table>"
                "<thead><tr>"
                "<th>Severidade</th>"
                "<th>Regra</th>"
                "<th>Linha</th>"
                "<th>Coluna</th>"
                "<th>Valor</th>"
                "<th>Mensagem</th>"
                "</tr></thead>"
                f"<tbody>{issue_rows}</tbody>"
                "</table>"
                "</div>"
            )
        else:
            issues_section = (
                "<p>Nenhum problema encontrado.</p>"
            )

        return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataAudit Pro - {html.escape(profile.name)}</title>
<style>
body {{
    margin: 0;
    background: #f5f6fa;
    color: #202333;
    font-family: Inter, Arial, sans-serif;
}}
.container {{
    max-width: 1240px;
    margin: auto;
    padding: 32px 20px 60px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}}
.card {{
    background: white;
    border: 1px solid #e4e7ef;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.05);
}}
.section {{
    margin-top: 22px;
}}
.label {{
    color: #6c7280;
    font-size: 13px;
}}
.value {{
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}}
.table-wrap {{
    overflow: auto;
    max-height: 520px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid #e8eaf0;
    font-size: 13px;
}}
th {{
    position: sticky;
    top: 0;
    background: #fafbfe;
}}
@media(max-width: 800px) {{
    .grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
</style>
</head>
<body>
<div class="container">
    <h1>DataAudit Pro</h1>
    <p>Auditoria da base <strong>{html.escape(profile.name)}</strong></p>

    <div class="grid">
        <div class="card">
            <div class="label">Linhas</div>
            <div class="value">{profile.rows}</div>
        </div>
        <div class="card">
            <div class="label">Colunas</div>
            <div class="value">{profile.columns}</div>
        </div>
        <div class="card">
            <div class="label">Erros</div>
            <div class="value">{errors}</div>
        </div>
        <div class="card">
            <div class="label">Avisos</div>
            <div class="value">{warnings}</div>
        </div>
    </div>

    <div class="card section">
        <h2>Perfil das colunas</h2>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Coluna</th>
                        <th>Tipo</th>
                        <th>Nulos</th>
                        <th>% nulos</th>
                        <th>Unicos</th>
                        <th>% unicos</th>
                        <th>Outliers IQR</th>
                    </tr>
                </thead>
                <tbody>
                    {profile_rows}
                </tbody>
            </table>
        </div>
    </div>

    <div class="card section">
        <h2>Problemas encontrados</h2>
        {issues_section}
    </div>
</div>
</body>
</html>
"""

    @staticmethod
    def build_reconciliation_html(
        summary: ReconciliationSummary,
        result: pd.DataFrame,
    ) -> str:
        divergent = result[
            result[
                "reconciliation_status"
            ]
            == "divergente"
        ].head(
            200
        )

        divergent_rows = ""

        for _, row in divergent.iterrows():
            identifier = html.escape(
                str(
                    row.get(
                        "id",
                        "",
                    )
                )
            )

            differences = html.escape(
                str(
                    row.get(
                        "differences",
                        "",
                    )
                )
            )

            divergent_rows += (
                "<tr>"
                f"<td>{identifier}</td>"
                f"<td>{differences}</td>"
                "</tr>"
            )

        if divergent_rows:
            divergent_section = (
                "<table>"
                "<thead>"
                "<tr>"
                "<th>ID</th>"
                "<th>Campos divergentes</th>"
                "</tr>"
                "</thead>"
                f"<tbody>{divergent_rows}</tbody>"
                "</table>"
            )
        else:
            divergent_section = (
                "<p>Nenhuma divergencia encontrada.</p>"
            )

        return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataAudit Pro - Conciliacao</title>
<style>
body {{
    margin: 0;
    background: #f5f6fa;
    color: #202333;
    font-family: Inter, Arial, sans-serif;
}}
.container {{
    max-width: 1180px;
    margin: auto;
    padding: 32px 20px 60px;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
}}
.card {{
    background: white;
    border: 1px solid #e4e7ef;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.05);
}}
.section {{
    margin-top: 22px;
}}
.label {{
    color: #6c7280;
    font-size: 13px;
}}
.value {{
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
}}
th, td {{
    text-align: left;
    padding: 10px;
    border-bottom: 1px solid #e8eaf0;
}}
@media(max-width: 800px) {{
    .grid {{
        grid-template-columns: repeat(2, 1fr);
    }}
}}
</style>
</head>
<body>
<div class="container">
    <h1>Conciliacao de bases</h1>
    <p>Resumo automatico das correspondencias e divergencias.</p>

    <div class="grid">
        <div class="card">
            <div class="label">Conciliados</div>
            <div class="value">{summary.perfect_matches}</div>
        </div>
        <div class="card">
            <div class="label">Com divergencia</div>
            <div class="value">{summary.matched_with_differences}</div>
        </div>
        <div class="card">
            <div class="label">Somente esquerda</div>
            <div class="value">{summary.left_only_rows}</div>
        </div>
        <div class="card">
            <div class="label">Somente direita</div>
            <div class="value">{summary.right_only_rows}</div>
        </div>
    </div>

    <div class="card section">
        <h2>Taxas de correspondencia</h2>
        <p>Base esquerda:
            <strong>{summary.match_rate_left_pct:.2f}%</strong>
        </p>
        <p>Base direita:
            <strong>{summary.match_rate_right_pct:.2f}%</strong>
        </p>
    </div>

    <div class="card section">
        <h2>Divergencias</h2>
        {divergent_section}
    </div>
</div>
</body>
</html>
"""


# =============================================================================
# GERADOR DE DADOS DE EXEMPLO
# =============================================================================

class SampleDataGenerator:
    @staticmethod
    def generate(
        output_dir: str | Path,
    ) -> None:
        output = ensure_output_dir(
            output_dir
        )

        rng = np.random.default_rng(
            42
        )

        size = 150

        ids = [
            f"V{i:05d}"
            for i in range(
                1,
                size + 1,
            )
        ]

        dates = pd.date_range(
            "2026-01-01",
            periods=size,
            freq="D",
        )

        clients = [
            "Empresa Aurora",
            "Mercado Central",
            "Loja Horizonte",
            "Grupo Atlas",
            "Comercial Rio",
            "Papelaria Sul",
            "Tech Norte",
        ]

        categories = [
            "software",
            "servicos",
            "varejo",
            "assinatura",
        ]

        statuses = [
            "pago",
            "pendente",
            "cancelado",
        ]

        system = pd.DataFrame(
            {
                "ID Transação": ids,
                "Data Venda": dates,
                "Nome Cliente": rng.choice(
                    clients,
                    size,
                ),
                "Categoria": rng.choice(
                    categories,
                    size,
                ),
                "Valor Total": np.round(
                    rng.uniform(
                        50,
                        6000,
                        size,
                    ),
                    2,
                ),
                "Situação": rng.choice(
                    statuses,
                    size,
                    p=[
                        0.72,
                        0.20,
                        0.08,
                    ],
                ),
            }
        )

        # Problemas propositais para demonstracao.
        system.loc[
            5,
            "Nome Cliente",
        ] = None

        system.loc[
            7,
            "Valor Total",
        ] = -400

        system.loc[
            12,
            "Situação",
        ] = "Em analise"

        system.loc[
            20,
            "ID Transação",
        ] = system.loc[
            19,
            "ID Transação",
        ]

        system.loc[
            33,
            "Valor Total",
        ] = 2500000

        system.loc[
            45,
            "Nome Cliente",
        ] = "  Empresa   Aurora "

        system.loc[
            60,
            "ID Transação",
        ] = "INVALIDO"

        finance = system.copy()

        finance = finance.drop(
            index=[
                4,
                44,
                104,
            ]
        ).reset_index(
            drop=True
        )

        finance.loc[
            10,
            "Valor Total",
        ] = (
            float(
                finance.loc[
                    10,
                    "Valor Total",
                ]
            )
            + 25.50
        )

        finance.loc[
            30,
            "Situação",
        ] = "pendente"

        finance.loc[
            50,
            "Nome Cliente",
        ] = "EMPRESA AURORA"

        extras = pd.DataFrame(
            {
                "ID Transação": [
                    "V99991",
                    "V99992",
                ],
                "Data Venda": [
                    "2026-08-01",
                    "2026-08-02",
                ],
                "Nome Cliente": [
                    "Cliente Extra 1",
                    "Cliente Extra 2",
                ],
                "Categoria": [
                    "servicos",
                    "varejo",
                ],
                "Valor Total": [
                    900.0,
                    1200.0,
                ],
                "Situação": [
                    "pago",
                    "pendente",
                ],
            }
        )

        finance = pd.concat(
            [
                finance,
                extras,
            ],
            ignore_index=True,
        )

        system_path = (
            output
            / "vendas_sistema.csv"
        )

        finance_path = (
            output
            / "vendas_financeiro.xlsx"
        )

        config_path = (
            output
            / "config.json"
        )

        system.to_csv(
            system_path,
            index=False,
            encoding="utf-8-sig",
        )

        finance.to_excel(
            finance_path,
            index=False,
        )

        ConfigManager.save_default(
            config_path
        )

        LOGGER.info(
            "Dados de demonstracao gerados em %s",
            output,
        )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="data_audit_pro",
        description=(
            "Auditoria, limpeza e conciliacao de arquivos CSV/Excel."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ativa logs detalhados.",
    )

    commands = parser.add_subparsers(
        dest="command",
        required=True,
    )

    audit = commands.add_parser(
        "audit",
        help="Audita uma unica base.",
    )

    audit.add_argument(
        "--input",
        required=True,
        help="Arquivo de entrada.",
    )

    audit.add_argument(
        "--config",
        help="Arquivo JSON de configuracao.",
    )

    audit.add_argument(
        "--output",
        default="./output",
        help="Pasta de saida.",
    )

    reconcile = commands.add_parser(
        "reconcile",
        help="Concilia duas bases.",
    )

    reconcile.add_argument(
        "--left",
        required=True,
        help="Base esquerda/principal.",
    )

    reconcile.add_argument(
        "--right",
        required=True,
        help="Base direita/secundaria.",
    )

    reconcile.add_argument(
        "--config",
        help="Arquivo JSON de configuracao.",
    )

    reconcile.add_argument(
        "--output",
        default="./output",
        help="Pasta de saida.",
    )

    sample = commands.add_parser(
        "sample",
        help="Gera dados de demonstracao.",
    )

    sample.add_argument(
        "--output",
        default="./demo",
        help="Pasta de destino.",
    )

    init_config = commands.add_parser(
        "init-config",
        help="Gera config.json inicial.",
    )

    init_config.add_argument(
        "--output",
        default="./config.json",
        help="Caminho do config.json.",
    )

    return parser


def run_audit(
    args: argparse.Namespace,
) -> None:
    config = ConfigManager.load(
        args.config
    )

    pipeline = AuditPipeline(
        config
    )

    cleaned, profile, issues = (
        pipeline.audit_file(
            args.input
        )
    )

    writer = ReportWriter(
        args.output
    )

    writer.write_audit(
        cleaned_df=cleaned,
        profile=profile,
        issues=issues,
        base_name=Path(
            args.input
        ).stem,
    )

    print(
        "\nAUDITORIA CONCLUIDA"
    )

    print(
        f"Linhas: {profile.rows}"
    )

    print(
        f"Colunas: {profile.columns}"
    )

    print(
        f"Problemas: {len(issues)}"
    )

    print(
        "Erros: "
        f"{sum(issue.severity == 'error' for issue in issues)}"
    )

    print(
        "Avisos: "
        f"{sum(issue.severity == 'warning' for issue in issues)}"
    )

    print(
        "Saida: "
        f"{Path(args.output).resolve()}"
    )


def run_reconcile(
    args: argparse.Namespace,
) -> None:
    config = ConfigManager.load(
        args.config
    )

    pipeline = AuditPipeline(
        config
    )

    (
        left_cleaned,
        left_profile,
        left_issues,
    ) = pipeline.audit_file(
        args.left,
        dataset_name="base_esquerda",
    )

    (
        right_cleaned,
        right_profile,
        right_issues,
    ) = pipeline.audit_file(
        args.right,
        dataset_name="base_direita",
    )

    writer = ReportWriter(
        args.output
    )

    writer.write_audit(
        left_cleaned,
        left_profile,
        left_issues,
        "base_esquerda",
    )

    writer.write_audit(
        right_cleaned,
        right_profile,
        right_issues,
        "base_direita",
    )

    reconciler = DataReconciler(
        config.get(
            "reconciliation",
            {},
        )
    )

    result, summary = reconciler.reconcile(
        left_cleaned,
        right_cleaned,
    )

    writer.write_reconciliation(
        result=result,
        summary=summary,
        left_profile=left_profile,
        right_profile=right_profile,
    )

    print(
        "\nCONCILIACAO CONCLUIDA"
    )

    print(
        f"Base esquerda: {summary.left_rows}"
    )

    print(
        f"Base direita: {summary.right_rows}"
    )

    print(
        f"Conciliados: {summary.perfect_matches}"
    )

    print(
        "Com divergencias: "
        f"{summary.matched_with_differences}"
    )

    print(
        "Somente esquerda: "
        f"{summary.left_only_rows}"
    )

    print(
        "Somente direita: "
        f"{summary.right_only_rows}"
    )

    print(
        "Saida: "
        f"{Path(args.output).resolve()}"
    )


def run_sample(
    args: argparse.Namespace,
) -> None:
    SampleDataGenerator.generate(
        args.output
    )

    print(
        "\nExemplos criados em: "
        f"{Path(args.output).resolve()}"
    )


def run_init_config(
    args: argparse.Namespace,
) -> None:
    path = Path(
        args.output
    )

    ConfigManager.save_default(
        path
    )

    print(
        "Configuracao criada em: "
        f"{path.resolve()}"
    )


def main() -> int:
    parser = build_parser()

    args = parser.parse_args()

    configure_logging(
        args.verbose
    )

    try:
        if args.command == "audit":
            run_audit(
                args
            )

        elif args.command == "reconcile":
            run_reconcile(
                args
            )

        elif args.command == "sample":
            run_sample(
                args
            )

        elif args.command == "init-config":
            run_init_config(
                args
            )

        else:
            parser.error(
                "Comando invalido."
            )

        return 0

    except KeyboardInterrupt:
        LOGGER.error(
            "Operacao cancelada pelo usuario."
        )
        return 130

    except Exception as exc:
        LOGGER.exception(
            "Falha: %s",
            exc,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
