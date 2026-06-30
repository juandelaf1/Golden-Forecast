# Golden Forecast - DataScope Solutions

## Contexto

Somos **DataScope Solutions**, consultora internacional de análisis de datos. Este proyecto aplica Machine Learning supervisado sobre datos históricos del oro para generar predicciones explicables.

## Equipo

| Rol | Nombre |
|-----|--------|
| Product Owner | María |
| Scrum Master | Juan |
| Development Team | José, Gema, Joel |

## Problema de negocio

Predecir el comportamiento del precio del oro (GLD) usando datos históricos. Abordamos el problema desde dos enfoques:

- **Regresión**: predecir el precio de cierre (valor numérico continuo)
- **Clasificación**: predecir si el precio subirá o bajará al día siguiente (binario)

## Dataset

Datos históricos del ETF **GLD** (Gold Trust) vía Yahoo Finance, desde 2010 hasta la actualidad. Variables principales: Open, High, Low, Close, Volume. Enriquecido con SP500 y USD Index como features externas.

## Stack tecnológico

Python · pandas · numpy · scikit-learn · matplotlib · seaborn · yfinance

## Estructura del repositorio

```
golden-forecast/
├── data/          # Datos crudos y procesados
├── notebooks/     # Notebooks del pipeline completo
├── src/           # Funciones reutilizables
├── models/        # Modelos entrenados (.pkl)
├── slides/        # Presentación final
├── docs/          # Documentación y decisiones
├── README.md
├── ROADMAP.md
├── requirements.txt
└── .gitignore
```

## Cómo ejecutar

1. Clonar el repositorio
2. `pip install -r requirements.txt`
3. Abrir y ejecutar los notebooks en orden numérico

## Licencia

Proyecto académico sin fines comerciales.
