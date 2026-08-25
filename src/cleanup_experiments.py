#!/usr/bin/env python3
"""
Limpeza segura de execuções antigas de experimentos.

Procura nomes contendo timestamps no formato:
    <experimento>_YYYYMMDD_HHMMSS

Exemplos:
    mlp_base_20260725_220146
    pinn_cont_base_20260725_054504
    pinn_cont_base_20260725_054504_predictions.parquet

Por segurança, o modo padrão é somente simulação. A exclusão real exige --apply.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RUN_PATTERN = re.compile(
    r"(?P<experiment>[A-Za-z0-9][A-Za-z0-9_.-]*?)_"
    r"(?P<timestamp>\d{8}_\d{6})"
    r"(?=$|[_\-.])"
)

DEFAULT_ROOTS = (
    Path("results/metrics"),
    Path("results/plots"),
    Path("results/models"),
    Path("logs"),
)

DEFAULT_PROTECTED_NAMES = {
    "final",
    "publication",
    "publications",
    "published",
    "best",
    "baseline_final",
    "archive",
    "archives",
}


@dataclass(frozen=True)
class RunKey:
    experiment: str
    timestamp_text: str

    @property
    def timestamp(self) -> datetime:
        return datetime.strptime(self.timestamp_text, TIMESTAMP_FORMAT)

    @property
    def run_id(self) -> str:
        return f"{self.experiment}_{self.timestamp_text}"


@dataclass
class RunArtifacts:
    key: RunKey
    paths: set[Path] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove versões antigas de experimentos mantendo as execuções "
            "mais recentes de cada grupo."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        type=Path,
        default=list(DEFAULT_ROOTS),
        help=(
            "Pastas varridas. Padrão: results/metrics results/plots "
            "results/models results/logs"
        ),
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Quantidade de execuções recentes mantidas por experimento.",
    )
    parser.add_argument(
        "--experiment",
        default="*",
        help='Filtro fnmatch. Exemplos: "pinn_cont_base", "pinn_*", "*base".',
    )
    parser.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Remove apenas execuções mais antigas que DAYS dias.",
    )
    parser.add_argument(
        "--protect",
        nargs="*",
        default=sorted(DEFAULT_PROTECTED_NAMES),
        help="Nomes de diretórios protegidos.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Executa a exclusão. Sem esta opção, apenas simula.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Não solicita confirmação interativa. Requer --apply.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra também as execuções preservadas.",
    )
    return parser.parse_args()


def extract_run_key(name: str) -> RunKey | None:
    match = RUN_PATTERN.search(name)
    if match is None:
        return None

    timestamp_text = match.group("timestamp")
    try:
        datetime.strptime(timestamp_text, TIMESTAMP_FORMAT)
    except ValueError:
        return None

    experiment = match.group("experiment").rstrip("._-")
    if not experiment:
        return None

    return RunKey(experiment=experiment, timestamp_text=timestamp_text)


def is_protected(path: Path, protected_names: set[str]) -> bool:
    protected_lower = {name.lower() for name in protected_names}
    return any(part.lower() in protected_lower for part in path.parts)


def scan_runs(
    roots: Iterable[Path],
    protected_names: set[str],
) -> dict[RunKey, RunArtifacts]:
    runs: dict[RunKey, RunArtifacts] = {}

    for root in roots:
        if not root.exists():
            print(f"[aviso] Raiz não encontrada, ignorando: {root}")
            continue
        if not root.is_dir():
            print(f"[aviso] A raiz não é diretório, ignorando: {root}")
            continue

        for current_root, dirnames, filenames in os.walk(root):
            current_path = Path(current_root)
            dirnames[:] = [
                dirname
                for dirname in dirnames
                if not is_protected(current_path / dirname, protected_names)
            ]

            candidates = [
                *(current_path / dirname for dirname in dirnames),
                *(current_path / filename for filename in filenames),
            ]

            for path in candidates:
                if is_protected(path, protected_names):
                    continue

                key = extract_run_key(path.name)
                if key is None:
                    continue

                run = runs.setdefault(key, RunArtifacts(key=key))
                run.paths.add(path)

    return runs


def select_runs_to_remove(
    runs: dict[RunKey, RunArtifacts],
    keep: int,
    experiment_pattern: str,
    older_than_days: int | None,
    now: datetime,
) -> tuple[list[RunArtifacts], list[RunArtifacts]]:
    grouped: dict[str, list[RunArtifacts]] = {}

    for run in runs.values():
        if not fnmatch.fnmatch(run.key.experiment, experiment_pattern):
            continue
        grouped.setdefault(run.key.experiment, []).append(run)

    remove: list[RunArtifacts] = []
    preserve: list[RunArtifacts] = []

    age_limit = (
        now - timedelta(days=older_than_days)
        if older_than_days is not None
        else None
    )

    for experiment_runs in grouped.values():
        ordered = sorted(
            experiment_runs,
            key=lambda item: item.key.timestamp,
            reverse=True,
        )

        for index, run in enumerate(ordered):
            inside_keep_window = index < keep
            old_enough = age_limit is None or run.key.timestamp < age_limit

            if not inside_keep_window and old_enough:
                remove.append(run)
            else:
                preserve.append(run)

    remove.sort(key=lambda item: (item.key.experiment, item.key.timestamp))
    preserve.sort(key=lambda item: (item.key.experiment, item.key.timestamp))
    return remove, preserve


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def collapse_nested_paths(paths: Iterable[Path]) -> list[Path]:
    normalized = sorted(
        {path.resolve(strict=False) for path in paths},
        key=lambda path: (len(path.parts), str(path)),
    )

    collapsed: list[Path] = []
    for path in normalized:
        if any(is_relative_to(path, parent) for parent in collapsed):
            continue
        collapsed.append(path)
    return collapsed


def path_size(path: Path) -> int:
    try:
        if path.is_symlink() or path.is_file():
            return path.stat().st_size
        if path.is_dir():
            total = 0
            for child in path.rglob("*"):
                try:
                    if child.is_file() or child.is_symlink():
                        total += child.stat().st_size
                except OSError:
                    continue
            return total
    except OSError:
        return 0
    return 0


def format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def print_plan(
    remove: list[RunArtifacts],
    preserve: list[RunArtifacts],
    verbose: bool,
    apply: bool,
) -> None:
    mode = "EXCLUSÃO REAL" if apply else "SIMULAÇÃO (dry-run)"

    print("\n" + "=" * 72)
    print(f"LIMPEZA DE EXPERIMENTOS — {mode}")
    print("=" * 72)

    if verbose and preserve:
        print("\nExecuções preservadas:")
        for run in preserve:
            print(f"  [manter] {run.key.run_id}")

    if not remove:
        print("\nNenhuma execução atende aos critérios de remoção.")
        print("=" * 72)
        return

    print("\nExecuções selecionadas para remoção:")
    total_size = 0
    total_paths = 0

    for run in remove:
        paths = collapse_nested_paths(run.paths)
        size = sum(path_size(path) for path in paths)
        total_size += size
        total_paths += len(paths)

        print(f"\n  [remover] {run.key.run_id} — {format_bytes(size)}")
        for path in paths:
            print(f"      {path}")

    print("\nResumo:")
    print(f"  Execuções:       {len(remove)}")
    print(f"  Caminhos:        {total_paths}")
    print(f"  Espaço estimado: {format_bytes(total_size)}")
    print("=" * 72)


def confirm_deletion() -> bool:
    answer = input(
        "\nDigite 'EXCLUIR' para confirmar a remoção permanente: "
    ).strip()
    return answer == "EXCLUIR"


def delete_path(path: Path) -> tuple[bool, str | None]:
    try:
        if not path.exists() and not path.is_symlink():
            return False, "caminho não existe"

        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            return False, "tipo de caminho não suportado"

        return True, None
    except OSError as exc:
        return False, str(exc)


def apply_cleanup(remove: list[RunArtifacts]) -> int:
    deleted = 0
    failures = 0

    all_paths = collapse_nested_paths(
        path for run in remove for path in run.paths
    )

    all_paths.sort(
        key=lambda path: (
            0 if path.is_file() or path.is_symlink() else 1,
            -len(path.parts),
            str(path),
        )
    )

    print("\nExecutando limpeza...")
    for path in all_paths:
        success, error = delete_path(path)
        if success:
            deleted += 1
            print(f"  [ok] {path}")
        else:
            failures += 1
            print(f"  [falha] {path}: {error}")

    print("\nResultado:")
    print(f"  Caminhos removidos: {deleted}")
    print(f"  Falhas:             {failures}")
    return 0 if failures == 0 else 2


def validate_args(args: argparse.Namespace) -> None:
    if args.keep < 0:
        raise ValueError("--keep deve ser maior ou igual a zero.")
    if args.older_than is not None and args.older_than < 0:
        raise ValueError("--older-than deve ser maior ou igual a zero.")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Erro de configuração: {exc}", file=sys.stderr)
        return 2

    roots = [path.expanduser() for path in args.roots]
    protected_names = set(args.protect)

    runs = scan_runs(roots=roots, protected_names=protected_names)

    remove, preserve = select_runs_to_remove(
        runs=runs,
        keep=args.keep,
        experiment_pattern=args.experiment,
        older_than_days=args.older_than,
        now=datetime.now(),
    )

    print_plan(
        remove=remove,
        preserve=preserve,
        verbose=args.verbose,
        apply=args.apply,
    )

    if not remove:
        return 0

    if not args.apply:
        print("\nNada foi excluído. Use --apply para executar a limpeza.")
        return 0

    if not args.yes and not confirm_deletion():
        print("Operação cancelada.")
        return 1

    return apply_cleanup(remove)


if __name__ == "__main__":
    raise SystemExit(main())
