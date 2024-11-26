"""
Path utilities.
"""

from pathlib import Path
from string import Template

def patify(config_dict: dict[str, str | int | float | Path]) -> dict[str, str | int | float | Path]:
    """
    Convert all string path values in the dictionary to Path objects.
    """
    for key, value in config_dict.items():
        if key.endswith('_path') and isinstance(value, str):
            config_dict[key] = Path(value)
    return config_dict

def unpatify(config_dict: dict[str, str | int | float | Path]) -> dict[str, str | int | float | str | Path]:
    """
    Convert all Path objects in the dictionary to string paths.
    """
    for key, value in config_dict.items():
        if key.endswith('_path') and isinstance(value, Path):
            config_dict[key] = str(value)
    return config_dict

def gen_from_template(template_path: Path, values: dict[str, str | int | float | Path], out_filepath: Path):
    """
    Generate a file from a template file.
    """
    with template_path.open('r', encoding='utf-8') as file_template:
        template: Template = Template(file_template.read())
        content: str = template.safe_substitute(values)
        with out_filepath.open('w', encoding='utf-8') as file_out:
            file_out.write(content)
