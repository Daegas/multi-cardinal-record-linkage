# multi-cardinal-record-linkage
Main repository for my master's thesis.


## Instalación

Clona o descarga el repositorio y entra en su directorio raíz.

```bash
git clone git@github.com:Daegas/multi-cardinal-record-linkage.git
cd multi-cardinal-record-linkage
```

(Recomendado) Crea un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instala la biblioteca en modo editable:

```bash
python3 -m pip install -e .
```


### Configuración de la ruta

La ubicación del directorio que contiene los datos se configura mediante un archivo `.env`.

Para configurarlo, copia el archivo `.env.example` y renombra la copia como `.env`:

```bash
cp .env-example .env
```

Abre el archivo `.env` y modifica la variable `DATA_DIR` para indicar la ubicación del directorio que contiene los datos.

Se pueden utilizar rutas relativas a la raíz del proyecto:

```bash
DATA_DIR=data/
```

O rutas absolutas:

```bash
DATA_DIR=/Users/usuario/Documents/mis_datos/
```

Dentro de DATA_DIR debe haber existir `cleaned/datasets/dataset_A.parquet`.
