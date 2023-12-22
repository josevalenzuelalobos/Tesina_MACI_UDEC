# Aplicación de Teledetección y Aprendizaje Automático para la Detección de Pérdida de Biomasa en Bosques Chilenos

Este repositorio contiene el código y los recursos utilizados en mi tesis de maestría en Ciencia de Datos. El proyecto se enfoca en la aplicación de técnicas de teledetección y aprendizaje automático para detectar la pérdida de biomasa en bosques chilenos.

## Descripción

El proyecto utiliza datos de satélite y algoritmos de aprendizaje automático para analizar y predecir áreas de pérdida de biomasa en bosques. Se emplean técnicas avanzadas para procesar y analizar grandes conjuntos de datos de imágenes satelitales, proporcionando insights significativos sobre el estado y la salud de los bosques chilenos.

## Estructura del Repositorio

El repositorio se organiza de la siguiente manera:

- `Tesina_General_Utils.py`: Contiene utilidades generales, como la clase Logger para registro de mensajes y funciones para manejo de directorios y visualización de estructuras de datos.
- `Tesina_Images_Utils.py`: Incluye funciones para la normalización y análisis de imágenes, así como la creación de máscaras a partir de polígonos.
- `Tesina_Indexs_Utils.py`: Define índices de vegetación, agua, fuego y nubes, y proporciona funciones para calcular estos índices a partir de conjuntos de datos de satélite.
- `Tesina_Maps_Utils.py`: Utilidades para trabajar con coordenadas geográficas y convertirlas a coordenadas de píxeles.
- `Tesina_Models_Utils.py`: Funciones para guardar y analizar modelos de aprendizaje automático, incluyendo la generación de metadatos.
- `Tesina_Sentinel_Utils.py`: Funciones específicas para trabajar con datos del satélite Sentinel-2, como la clasificación de fechas soleadas y nubladas.
- `Jupyter Notebooks`: Contienen ejemplos prácticos y análisis detallados de los datos y modelos.
- `Scripts de Desarrollo`: Scripts utilizados durante el desarrollo del proyecto, que incluyen análisis exploratorios y pruebas de algoritmos.

## Requisitos

Para ejecutar este proyecto, es necesario instalar varias librerías de Python. Puedes instalar todas las dependencias necesarias ejecutando el siguiente comando en tu terminal:

```bash
pip install -r requirements.txt
```

## Cómo Usar

Para utilizar este código en tu propio proyecto o investigación, sigue estos pasos:

1. Clona el repositorio en tu máquina local.
2. Instala las dependencias necesarias (lista proporcionada en `requirements.txt`).
3. Ejecuta los scripts según sea necesario, modificando los parámetros y rutas de archivos según tus datos.

## Contribuciones y Contacto

Si tienes sugerencias o deseas contribuir a este proyecto, no dudes en abrir un 'issue' o un 'pull request'. Para consultas más específicas, puedes contactarme directamente en jose.valenzuela@wolke.cl.

## Licencia

Este proyecto está bajo la Licencia GPL. Consulta el archivo `LICENSE` para más detalles.
