"""Copy versioned, payload-free catalogs into the wheel; no raw data is packaged."""

from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithCatalogs(build_py):
    def run(self):
        super().run()
        target = Path(self.build_lib) / "vllm_hust_workload_data" / "resources"
        target.mkdir(parents=True, exist_ok=True)
        for source in (Path(__file__).parent / "data" / "catalogs").glob("*.json"):
            self.copy_file(str(source), str(target / source.name))
        for source in (Path(__file__).parent / "schemas").glob("*.json"):
            self.copy_file(str(source), str(target / source.name))
        self.copy_file(
            str(Path(__file__).parent / "MIGRATION_PROVENANCE.json"),
            str(target / "migration_provenance.json"),
        )


setup(cmdclass={"build_py": BuildWithCatalogs})
