# Tests base con pytest

Incluye una base mínima y útil para el repositorio `transport-ml-rd`:

- `tests/conftest.py`: agrega `src/` al `PYTHONPATH`
- `tests/test_utils.py`: pruebas de utilidades y transformaciones
- `tests/test_pipeline.py`: pruebas de configuración, separación de variables, construcción del modelo y métricas

## Uso

```bash
pytest
```
