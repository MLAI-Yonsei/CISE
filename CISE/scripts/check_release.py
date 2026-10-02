from pathlib import Path
import hashlib
import json
import re
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "configs", "docs", "provenance", "scripts", "tests", ".github")
ROOT_FILES = {"README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "pyproject.toml", "MANIFEST.in", ".gitignore"}
CONFIG_FILES = {"cci.json", "cci.demo.json", "llema.json", "llema.demo.json"}
IGNORED = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}
FORBIDDEN_PARTS = {".local", ".runtime", "runs", "assets", "data", "datasets", "opk"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".pickle", ".joblib", ".upf", ".csv", ".parquet", ".npy", ".npz", ".cif", ".jsonl", ".zip", ".log", ".pem", ".key"}
PATTERNS = (
    rb"/(?:data\d*|home|Users)/[A-Za-z0-9_.-]+/",
    rb"sk-or-v1-[A-Za-z0-9]{32,}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"gh[pousr]_[A-Za-z0-9]{30,}",
    rb"github_pat_[A-Za-z0-9_]{40,}",
    rb"sk-[A-Za-z0-9_-]{32,}",
    rb"AKIA[A-Z0-9]{16}",
)


def check_item(name, data):
    path = Path(name)
    if any(part in FORBIDDEN_PARTS for part in path.parts):
        raise ValueError(f"Excluded asset path: {name}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name in {"AK.md", ".env"} or path.name.startswith(".env."):
        raise ValueError(f"Excluded file: {name}")
    if any(re.search(pattern, data) for pattern in PATTERNS):
        raise ValueError(f"Private path or credential pattern: {name}")


def check_source():
    count = 0
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink not permitted: {relative}")
        if not path.is_file():
            continue
        if relative.parts[0] not in SOURCE_DIRS and str(relative) not in ROOT_FILES:
            raise ValueError(f"Unexpected public file: {relative}")
        if relative.parts[0] in SOURCE_DIRS:
            section = relative.parts[0]
            allowed = (
                (section in {"src", "tests", "scripts"} and path.suffix == ".py")
                or (section == "docs" and path.suffix == ".md")
                or (section == "configs" and len(relative.parts) == 2 and path.name in CONFIG_FILES)
                or (section == "provenance" and str(relative) == "provenance/sources.json")
                or (section == ".github" and path.suffix in {".yml", ".yaml"})
            )
            if not allowed:
                raise ValueError(f"Unexpected public asset: {relative}")
        check_item(str(relative), path.read_bytes())
        count += 1
    manifest = json.loads((ROOT / "provenance/sources.json").read_text())
    for row in manifest["sources"]:
        actual = hashlib.sha256((ROOT / row["destination"]).read_bytes()).hexdigest()
        if actual != row["distributed_sha256"]:
            raise ValueError(f"Provenance mismatch: {row['destination']}")
    print(f"Source: {count} files; asset, credential-pattern and provenance checks passed")


def check_archive_provenance(items, wheel=False):

    files = dict(items)
    expected = {row["destination"]: row["distributed_sha256"]
                for row in json.loads((ROOT / "provenance/sources.json").read_text())["sources"]}
    expected.update({str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (ROOT / "src").rglob("*.py")})
    for destination, digest in expected.items():
        target = destination
        if wheel:
            target = target.removeprefix("src/")
        matches = [data for name, data in files.items() if name == target or name.endswith("/" + target)]
        if len(matches) != 1 or hashlib.sha256(matches[0]).hexdigest() != digest:
            raise ValueError(f"Archive provenance mismatch: {target}")


def main():
    check_source()
    wheels = sorted((ROOT / "dist").glob("*.whl"))
    sdists = sorted((ROOT / "dist").glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Build exactly one wheel and one sdist in dist/ first")
    for archive in wheels + sdists:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as handle:
                items = [(name, handle.read(name)) for name in handle.namelist() if not name.endswith("/")]
        else:
            with tarfile.open(archive) as handle:
                if any(member.issym() or member.islnk() for member in handle.getmembers()):
                    raise ValueError("Archive links are not permitted")
                items = [(member.name, handle.extractfile(member).read()) for member in handle.getmembers() if member.isfile()]
        for name, data in items:
            check_item(name, data)
        check_archive_provenance(items, wheel=archive.suffix == ".whl")
        print(f"{archive.name}: {len(items)} files; asset and credential-pattern checks passed")


if __name__ == "__main__":
    main()
