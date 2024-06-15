import os
import fitz
import json
import random
import pandas as pd
import ast
import locale
from collections import Counter
from datetime import datetime
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value
from transformers import (AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification)
import numpy as np
import evaluate
import re
from heapq import nlargest
from io import BytesIO
import streamlit as st
import os

input_directory = "Data/facturas"
output_directory_plano = "Data/facturas_texto"


# Función para eliminar archivos en un directorio
def eliminar_archivos_en_directorio(directorio):
    archivos = os.listdir(directorio)
    for archivo in archivos:
        ruta_archivo = os.path.join(directorio, archivo)
        try:
            if os.path.isfile(ruta_archivo):
                os.remove(ruta_archivo)
            elif os.path.isdir(ruta_archivo):
                os.rmdir(ruta_archivo)
            else:
                st.warning(f'No se pudo identificar el tipo de archivo o directorio: {archivo}')
        except Exception as e:
            st.error(f'Error al eliminar "{archivo}": {e}')

# Eliminar archivos en el directorio de entrada
eliminar_archivos_en_directorio(input_directory)
eliminar_archivos_en_directorio(output_directory_plano)

# Crear directorio si no existe
os.makedirs(input_directory, exist_ok=True)

# Título y descripción de la aplicación
st.title('Extracción automática de texto estructurado de facturas')

# Widget para cargar archivos PDF
archivos_pdf = st.file_uploader("Selecciona facturas", type="pdf", accept_multiple_files=True)

# Procesar archivos PDF cargados
if archivos_pdf is not None and len(archivos_pdf) > 0:
    # Llamar a la función para eliminar archivos en el directorio de entrada
    eliminar_archivos_en_directorio(input_directory)
    eliminar_archivos_en_directorio(output_directory_plano)
    datos_facturas =[]
    # Inicializar una lista para almacenar los datos de las facturas
    if 'datos_facturas' in locals():
        datos_facturas.clear()
    else:
        datos_facturas = []

    # Guardar los archivos en el directorio de entrada y extraer datos
    for archivo in archivos_pdf:
        nombre_archivo = archivo.name
        ruta_guardado = os.path.join(input_directory, nombre_archivo)

        # Guardar el archivo en el directorio de entrada
        with open(ruta_guardado, 'wb') as f:
            f.write(archivo.getbuffer())

        # Aquí iría el código para extraer texto de los archivos PDF y procesarlo
        # Supongamos que extraes texto y lo guardas en una lista llamada texto_extraido
        texto_extraido = ["Ejemplo de texto extraído"]  # Aquí debes insertar el texto extraído

        # Crear un diccionario con los datos de la factura
        datos_factura = {
            'nombre_archivo': nombre_archivo,
            'texto': texto_extraido
        }

        # Agregar datos de la factura a la lista
        datos_facturas.append(datos_factura)

    # Crear DataFrame con los datos de las facturas
    df = pd.DataFrame(datos_facturas)













    def extraer_texto(pdf_path):
        texto = ""
        with fitz.open(pdf_path) as pdf:
            for page_num in range(len(pdf)):
                texto += pdf[page_num].get_text()
        return texto

    # Recorrer los archivos PDF en el directorio de entrada
    for filename in os.listdir(input_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_directory, filename)
            # Extraer texto del PDF
            texto_extraido = extraer_texto(pdf_path)
            # Guardar texto en formato plano
            with open(os.path.join(output_directory_plano, filename + ".txt"), "w", encoding="utf-8") as f:
                f.write(texto_extraido)


    # Definir la ruta de la carpeta
    carpeta = "Data/facturas_texto"

    # Verificar que la carpeta existe
    if not os.path.exists(carpeta):
        print(f"Error: La carpeta {carpeta} no existe.")
    else:
        # Obtener la lista de archivos en la carpeta
        archivos = os.listdir(carpeta)

        # Iterar sobre cada archivo en la carpeta
        for archivo in archivos:
            # Verificar si el archivo tiene la extensión .pdf.txt
            if archivo.endswith('.pdf.txt'):
                # Generar el nuevo nombre de archivo eliminando la parte .pdf
                nuevo_nombre = archivo.replace('.pdf', '')

                # Ruta completa del archivo original y nuevo
                ruta_original = os.path.join(carpeta, archivo)
                ruta_nueva = os.path.join(carpeta, nuevo_nombre)

                try:
                    # Renombrar el archivo
                    os.replace(ruta_original, ruta_nueva)
                    print(f"Archivo renombrado de {ruta_original} a {ruta_nueva}")
                except FileNotFoundError:
                    print(f"Error: El archivo {ruta_original} no existe.")
                except PermissionError:
                    print(f"Error: No tienes permisos para renombrar el archivo {ruta_original}.")
                except Exception as e:
                    print(f"Error al renombrar el archivo {ruta_original}: {e}")

    # Definir las rutas de las carpetas (sin cambio)
    txt_folder = "Data/facturas_texto"


    # json_folder = "Data/test"

    # Función para leer el contenido de un archivo de texto
    def read_txt_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


    # Lista para almacenar los datos de cada factura
    data = []

    # Iterar sobre los archivos de texto
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith(".txt"):
            txt_path = os.path.join(txt_folder, txt_file)
            # json_file = txt_file.replace(".txt", ".json")
            # json_path = os.path.join(json_folder, json_file)

            # Leer el contenido del archivo de texto
            text_content = read_txt_file(txt_path)

            # Leer el contenido del archivo JSON
            # with open(json_path, 'r', encoding='utf-8') as json_file:
            # json_data = json.load(json_file)

            # Obtener el ID del nombre del archivo
            file_id = os.path.splitext(txt_file)[0]

            # Agregar los datos al listado
            data.append({
                'id': file_id,
                'texto': text_content,
                # 'ner_tags': [],
                # 'json': json_data
            })

    # Convertir la lista de datos a un DataFrame de pandas
    df = pd.DataFrame(data)

    # Aplicar split() a la columna "texto"
    df['texto'] = df['texto'].apply(lambda x: x.split())

    def separar_puntuacion(lista_palabras):
        palabras_nuevas = []
        for palabra in lista_palabras:
            # Verificar si la palabra comienza con un paréntesis de apertura
            if palabra.startswith("(") and len(palabra) > 1:
                palabras_nuevas.append(palabra[0])  # Agregar el paréntesis de apertura
                palabra = palabra[1:]  # Remover el paréntesis de apertura para las siguientes verificaciones

            # Verificar si la palabra termina con puntuación o paréntesis de cierre
            if palabra.endswith((".", ",", ";", ":", ")", "?")) and len(palabra) > 1:
                palabras_nuevas.append(palabra[:-1])
                palabras_nuevas.append(palabra[-1])
            else:
                palabras_nuevas.append(palabra)
        return palabras_nuevas

    df['texto'] = df['texto'].apply(separar_puntuacion)

    # Cambiar el nombre de la columna 'tokens' a 'texto'
    df.rename(columns={'texto': 'tokens'}, inplace=True)


    #Ruta para guardar el archivo CSV
    csv_path = "Data/dataset_facturas.csv"

    # Guardar el DataFrame como un archivo CSV
    df.to_csv(csv_path, index=False)#

    print(f"El dataset se ha guardado en {csv_path}")

    # Convertir a Dataset de HuggingFace
    dataset_unlabeled = Dataset.from_pandas(df)

    if '__index_level_0__' in dataset_unlabeled.column_names:
        dataset_unlabeled = dataset_unlabeled.remove_columns(['__index_level_0__'])

    features_unlabeled = Features({
        'id': Value(dtype='string'),
        'tokens': Sequence(Value(dtype='string'))
    })

    dataset_unlabeled = dataset_unlabeled.cast(features_unlabeled)

    # Definir la lista de etiquetas NER
    label_list = ['O', 'B-NOM', 'I-NOM', 'B-DNI', 'I-DNI', 'B-CAL', 'I-CAL', 'B-CP', 'I-CP', 'B-LOC', 'I-LOC', 'B-PRO', 'I-PRO',
                  'B-NOMC', 'I-NOMC', 'B-CIF', 'I-CIF', 'B-DIRC', 'I-DIRC', 'B-CPC', 'I-CPC', 'B-LOCC', 'I-LOCC', 'B-PROC', 'I-PROC',
                  'B-NUMF', 'I-NUMF', 'B-INI', 'I-INI', 'B-FIN', 'I-FIN', 'B-FAC', 'I-FAC', 'B-CAR','I-CAR', 'B-PER',
                  'I-PER', 'B-POT', 'I-POT']


    # Cargar el tokenizer y el modelo guardado
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/distilbert-base-multi-cased-finetuned-typo-detection")
    model_path = "./model_15_035_chiquito"
    model = AutoModelForTokenClassification.from_pretrained(model_path, num_labels=len(label_list))

    # Tokenizar y alinear las etiquetas
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=512)
        return tokenized_inputs

    tokenized_dataset_unlabeled = dataset_unlabeled.map(tokenize_and_align_labels, batched=True)

    # Definir el data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir='./Data/output',  # Directorio donde se guardarán los resultados
        per_device_eval_batch_size=16,
        logging_dir='./Data/logs',  # Directorio para guardar los logs
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Realizar predicciones en el conjunto no etiquetado
    predictions, _, _ = trainer.predict(tokenized_dataset_unlabeled)
    predictions = np.argmax(predictions, axis=2)

    # Convertir las predicciones a nombres de etiquetas
    true_predictions = [
        [label_list[p] for p in prediction]
        for prediction in predictions
    ]

    # Convertir tokenized_dataset_unlabeled a DataFrame
    unlabeled_df = tokenized_dataset_unlabeled.to_pandas()

    # Asegurarse de que las columnas necesarias están presentes en el DataFrame
    if 'input_ids' in unlabeled_df.columns:
        unlabeled_df['tokens'] = unlabeled_df['input_ids'].apply(lambda x: tokenizer.convert_ids_to_tokens(x))

    # Agregar las predicciones (true_predictions) al DataFrame
    unlabeled_df['predict'] = true_predictions

    # Mostrar las primeras filas del DataFrame para verificar

    # Guardar los resultados en un nuevo archivo CSV
    #output_path = "/content/drive/MyDrive/dataset_con_predicciones.csv"
    #unlabeled_df.to_csv(output_path, index=False)

    # Crear una nueva columna 'id_num' con los números extraídos de la columna 'id'
    unlabeled_df['id_num'] = unlabeled_df['id'].apply(lambda x: int(x.split('_')[-1]))

    # Ordenar el DataFrame por la nueva columna 'id_num'
    unlabeled_df = unlabeled_df.sort_values(by='id_num')

    # Eliminar la columna 'id_num' si ya no es necesaria
    unlabeled_df = unlabeled_df.drop(columns=['id_num'])

    # Reseteamos el índice
    unlabeled_df = unlabeled_df.reset_index()

    # Verificar filas con longitudes desiguales en 'tokens' y 'predict'
    filas_con_errores = []

    for index, row in unlabeled_df.iterrows():
        if len(row['tokens']) != len(row['predict']):
            filas_con_errores.append(index)

    # Mostrar las filas con errores
    print("Filas con longitudes desiguales en 'tokens' y 'predict':", filas_con_errores)

    def unir_puntuacion(lista_palabras):
        palabras_originales = []
        buffer = ''

        for palabra in lista_palabras:
            if palabra.isalnum():
                if buffer:
                    if palabras_originales:
                        palabras_originales[-1] += buffer
                    else:
                        palabras_originales.append(buffer)
                    buffer = ''
                palabras_originales.append(palabra)
            else:
                buffer += palabra

        if buffer:
            if palabras_originales:
                palabras_originales[-1] += buffer
            else:
                palabras_originales.append(buffer)

        return palabras_originales

    # Agregar una columna 'json_comprobacion' con diccionario vacío
    unlabeled_df['json_comprobacion'] = [{} for _ in range(len(unlabeled_df))]

    def comprobar_etiquetas(df):
        for index, row in df.iterrows():
            # Inicializar listas para almacenar las ocurrencias de las diferentes entidades como cadena
            ocurrencias_nombre_cliente = []
            ocurrencias_dni_cliente = []
            ocurrencias_calle_cliente = []
            ocurrencias_cp_cliente = []
            ocurrencias_población_cliente = []
            ocurrencias_provincia_cliente = []
            ocurrencias_nombre_comercializadora = []
            ocurrencias_cif_comercializadora = []
            ocurrencias_dirección_comercializadora = []
            ocurrencias_cp_comercializadora = []
            ocurrencias_población_comercializadora = []
            ocurrencias_provincia_comercializadora = []
            ocurrencias_número_factura = []
            ocurrencias_inicio_periodo = []
            ocurrencias_fin_periodo = []
            ocurrencias_importe_factura = []
            ocurrencias_fecha_cargo = []
            ocurrencias_consumo_periodo = []
            ocurrencias_potencia_contratada = []

            # Inicializar listas para almacenar las palabras de las diferentes entidades
            nombre_cliente = []
            dni_cliente = []
            calle_cliente = []
            cp_cliente = []
            población_cliente = []
            provincia_cliente = []
            nombre_comercializadora = []
            cif_comercializadora = []
            dirección_comercializadora = []
            cp_comercializadora = []
            población_comercializadora = []
            provincia_comercializadora = []
            número_factura = []
            inicio_periodo = []
            fin_periodo = []
            importe_factura = []
            fecha_cargo = []
            consumo_periodo = []
            potencia_contratada = []

            # Iterar sobre las etiquetas para buscar secuencias correspondientes a diferentes entidades
            for i, etiqueta in enumerate(row['predict']):
                if etiqueta == 'B-NOM':
                    # Reiniciar la lista de palabras del nombre del cliente para cada nueva ocurrencia
                    nombre_cliente = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-NOM' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-NOM':
                            # Agregar la palabra al nombre del cliente
                            nombre_cliente.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-NOM'
                    # Agregar la ocurrencia del nombre del cliente como cadena a la lista
                    ocurrencias_nombre_cliente.append(' '.join(unir_puntuacion(nombre_cliente)))

                elif etiqueta == 'B-DNI':
                    # Reiniciar la lista de palabras del DNI del cliente para cada nueva ocurrencia
                    dni_cliente = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-DNI' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-DNI':
                            # Agregar la palabra al DNI del cliente
                            dni_cliente.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-DNI'
                    # Agregar la ocurrencia del DNI del cliente como cadena a la lista
                    ocurrencias_dni_cliente.append(' '.join(unir_puntuacion(dni_cliente)))

                elif etiqueta == 'B-CAL':
                    # Reiniciar la lista de palabras de la calle del cliente para cada nueva ocurrencia
                    calle_cliente = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-CAL' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-CAL':
                            # Agregar la palabra a la calle del cliente
                            calle_cliente.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-CAL'
                    # Agregar la ocurrencia de la calle del cliente como cadena a la lista
                    ocurrencias_calle_cliente.append(' '.join(unir_puntuacion(calle_cliente)))

                elif etiqueta == 'B-CP':
                    # Reiniciar la lista de palabras del código postal del cliente para cada nueva ocurrencia
                    cp_cliente = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-CP' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-CP':
                            # Agregar la palabra al código postal del cliente
                            cp_cliente.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-CP'
                    # Agregar la ocurrencia del código postal del cliente como cadena a la lista
                    ocurrencias_cp_cliente.append(' '.join(unir_puntuacion(cp_cliente)))

                elif etiqueta == 'B-LOC':
                    # Reiniciar la lista de palabras de la población del cliente para cada nueva ocurrencia
                    población_cliente = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-LOC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-LOC':
                            # Agregar la palabra a la población del cliente
                            población_cliente.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-LOC'
                    # Agregar la ocurrencia de la población del cliente como cadena a la lista
                    ocurrencias_población_cliente.append(' '.join(unir_puntuacion(población_cliente)))

                elif etiqueta == 'B-PRO':
                    provincia_cliente = [row['tokens'][i]]
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-PRO':
                            provincia_cliente.append(row['tokens'][j])
                        else:
                            break
                    ocurrencias_provincia_cliente.append(' '.join(unir_puntuacion(provincia_cliente)))

                elif etiqueta == 'B-NOMC':
                    # Reiniciar la lista de palabras del nombre de la comercializadora para cada nueva ocurrencia
                    nombre_comercializadora = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-NOMC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-NOMC':
                            # Agregar la palabra al nombre de la comercializadora
                            nombre_comercializadora.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-NOMC'
                    # Agregar la ocurrencia del nombre de la comercializadora como cadena a la lista
                    ocurrencias_nombre_comercializadora.append(' '.join(unir_puntuacion(nombre_comercializadora)))

                elif etiqueta == 'B-CIF':
                    # Reiniciar la lista de palabras del CIF de la comercializadora para cada nueva ocurrencia
                    cif_comercializadora = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-CIF' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-CIF':
                            # Agregar la palabra al CIF de la comercializadora
                            cif_comercializadora.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-CIF'
                    # Agregar la ocurrencia del CIF de la comercializadora como cadena a la lista
                    ocurrencias_cif_comercializadora.append(' '.join(unir_puntuacion(cif_comercializadora)))

                elif etiqueta == 'B-DIRC':
                    # Reiniciar la lista de palabras de la dirección de la comercializadora para cada nueva ocurrencia
                    dirección_comercializadora = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-DIRC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-DIRC':
                            # Agregar la palabra a la dirección de la comercializadora
                            dirección_comercializadora.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-DIRC'
                    # Agregar la ocurrencia de la dirección de la comercializadora como cadena a la lista
                    ocurrencias_dirección_comercializadora.append(' '.join(unir_puntuacion(dirección_comercializadora)))

                elif etiqueta == 'B-CPC':
                    # Reiniciar la lista de palabras del código postal de la comercializadora para cada nueva ocurrencia
                    cp_comercializadora = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-CPC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-CPC':
                            # Agregar la palabra al código postal de la comercializadora
                            cp_comercializadora.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-CPC'
                    # Agregar la ocurrencia del código postal de la comercializadora como cadena a la lista
                    ocurrencias_cp_comercializadora.append(' '.join(unir_puntuacion(cp_comercializadora)))

                elif etiqueta == 'B-LOCC':
                    # Reiniciar la lista de palabras de la población de la comercializadora para cada nueva ocurrencia
                    población_comercializadora = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-LOCC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-LOCC':
                            # Agregar la palabra a la población de la comercializadora
                            población_comercializadora.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-LOCC'
                    # Agregar la ocurrencia de la población de la comercializadora como cadena a la lista
                    ocurrencias_población_comercializadora.append(' '.join(unir_puntuacion(población_comercializadora)))

                elif etiqueta == 'B-PROC':
                    provincia_comercializadora = [row['tokens'][i]]
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-PROC':
                            provincia_comercializadora.append(row['tokens'][j])
                        else:
                            break
                    ocurrencias_provincia_comercializadora.append(' '.join(unir_puntuacion(provincia_comercializadora)))

                elif etiqueta == 'B-NUMF':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    número_factura = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-NUMF' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-NUMF':
                            # Agregar la palabra al número de factura
                            número_factura.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-NUMF'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_número_factura.append(' '.join(unir_puntuacion(número_factura)))

                elif etiqueta == 'B-INI':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    inicio_periodo = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-INI' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-INI':
                            # Agregar la palabra al número de factura
                            inicio_periodo.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-INI'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_inicio_periodo.append(' '.join(unir_puntuacion(inicio_periodo)))

                elif etiqueta == 'B-FIN':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    fin_periodo = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-FIN' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-FIN':
                            # Agregar la palabra al número de factura
                            fin_periodo.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-FIN'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_fin_periodo.append(' '.join(unir_puntuacion(fin_periodo)))

                elif etiqueta == 'B-FAC':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    importe_factura = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-FAC' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-FAC':
                            # Agregar la palabra al número de factura
                            importe_factura.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-NUMF'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_importe_factura.append(' '.join(unir_puntuacion(importe_factura)))

                elif etiqueta == 'B-CAR':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    fecha_cargo = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-NUMF' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-CAR':
                            # Agregar la palabra al número de factura
                            fecha_cargo.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-NUMF'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_fecha_cargo.append(' '.join(unir_puntuacion(fecha_cargo)))

                elif etiqueta == 'B-PER':
                    # Reiniciar la lista de palabras del número de factura para cada nueva ocurrencia
                    consumo_periodo = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-PER' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-PER':
                            # Agregar la palabra al número de factura
                            consumo_periodo.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-PER'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_consumo_periodo.append(' '.join(unir_puntuacion(consumo_periodo)))

                elif etiqueta == 'B-POT':
                    # Reiniciar la lista de palabras de potencia_contatada para cada nueva ocurrencia
                    potencia_contratada = [row['tokens'][i]]
                    # Buscar si hay etiquetas 'I-POT' después
                    for j in range(i+1, len(row['predict'])):
                        if row['predict'][j] == 'I-POT':
                            # Agregar la palabra al número de factura
                            potencia_contratada.append(row['tokens'][j])
                        else:
                            break  # Salir del bucle cuando no haya más etiquetas 'I-POT'
                    # Agregar la ocurrencia del número de factura como cadena a la lista
                    ocurrencias_potencia_contratada.append(' '.join(unir_puntuacion(potencia_contratada)))

            # Agregar las ocurrencias al DataFrame bajo la columna 'json_comprobacion'
            df.at[index, 'json_comprobacion']['nombre_cliente'] = ocurrencias_nombre_cliente
            df.at[index, 'json_comprobacion']['dni_cliente'] = ocurrencias_dni_cliente
            df.at[index, 'json_comprobacion']['calle_cliente'] = ocurrencias_calle_cliente
            df.at[index, 'json_comprobacion']['cp_cliente'] = ocurrencias_cp_cliente
            df.at[index, 'json_comprobacion']['población_cliente'] = ocurrencias_población_cliente
            df.at[index, 'json_comprobacion']['provincia_cliente'] = ocurrencias_provincia_cliente  # Agregar provincia_cliente
            df.at[index, 'json_comprobacion']['nombre_comercializadora'] = ocurrencias_nombre_comercializadora
            df.at[index, 'json_comprobacion']['cif_comercializadora'] = ocurrencias_cif_comercializadora
            df.at[index, 'json_comprobacion']['dirección_comercializadora'] = ocurrencias_dirección_comercializadora
            df.at[index, 'json_comprobacion']['cp_comercializadora'] = ocurrencias_cp_comercializadora
            df.at[index, 'json_comprobacion']['población_comercializadora'] = ocurrencias_población_comercializadora
            df.at[index, 'json_comprobacion']['provincia_comercializadora'] = ocurrencias_provincia_comercializadora  # Agregar provincia_comercializadora
            df.at[index, 'json_comprobacion']['número_factura'] = ocurrencias_número_factura
            df.at[index, 'json_comprobacion']['inicio_periodo'] = ocurrencias_inicio_periodo
            df.at[index, 'json_comprobacion']['fin_periodo'] = ocurrencias_fin_periodo
            df.at[index, 'json_comprobacion']['importe_factura'] = ocurrencias_importe_factura
            df.at[index, 'json_comprobacion']['fecha_cargo'] = ocurrencias_fecha_cargo
            df.at[index, 'json_comprobacion']['consumo_periodo'] = ocurrencias_consumo_periodo
            df.at[index, 'json_comprobacion']['potencia_contratada'] = ocurrencias_potencia_contratada


        return df

    comprobar_etiquetas(unlabeled_df)

    def reconstruct_entities_combined(dictionary):
        reconstructed_dict = {}

        for key, entities in dictionary.items():
            # Paso 1: Eliminar "[UNK]" y reconstruir entidades basadas en "##"
            reconstructed = []
            current_entity = ""

            for entity in entities:
                # Eliminar los "[UNK]"
                entity = entity.replace("[UNK]", "")

                if "##" in entity:
                    entity = entity.replace("##", "")
                    current_entity += entity
                elif any(punctuation in entity for punctuation in ["-", ",", ".", ":", ";", "!", "?", "/"]):
                    current_entity += entity
                else:
                    if current_entity:
                        reconstructed.append(current_entity)
                    current_entity = entity

            if current_entity:
                reconstructed.append(current_entity)

            # Paso 2: Verificar y reconstruir cadenas que terminan en signos de puntuación
            entities = reconstructed
            reconstructed = []
            while entities:
                if entities[-1].endswith(("-", ",", ".", ":", ";", "!", "?", "/")):
                    reconstructed.append(entities.pop())
                else:
                    entity = entities.pop()
                    while entities and entities[-1].endswith(("-", ",", ".", ":", ";", "!", "?", "/")):
                        entity = entities.pop() + entity
                    reconstructed.append(entity)

            reconstructed.reverse()

            # Paso 3: Reconstruir entidades basadas en las últimas seis letras repetidas
            entities = reconstructed
            reconstructed = []

            for entity in entities:
                last_six = entity[-6:]
                if entity.count(last_six) > 1:
                    first_occurrence_index = entity.index(last_six)
                    reconstructed.append(entity[:first_occurrence_index + len(last_six)])
                else:
                    reconstructed.append(entity)

            reconstructed_dict[key] = reconstructed

        return reconstructed_dict


    unlabeled_df['json_comprobacion'] = unlabeled_df['json_comprobacion'].apply(reconstruct_entities_combined)

    def select_longest_string(d):
        # Iterar sobre el diccionario
        for key, value in d.items():
            # Si el valor es una lista de strings
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                # Seleccionar el string más largo y, en caso de empate, el primero en la lista
                longest_string = max(value, key=lambda x: (len(x), value.index(x)), default=None)
                # Actualizar el valor de la clave con el string más largo
                d[key] = [longest_string] if longest_string else []
        return d

    # Aplicar la función select_longest_string a la columna 'json_comprobacion' del DataFrame
    unlabeled_df['json_comprobacion'] = unlabeled_df['json_comprobacion'].apply(select_longest_string)


    # Cargar el DataFrame
    csv_path = "./Data/dataset_facturas.csv"
    df = pd.read_csv(csv_path)


    # Convertir las cadenas de tokens a listas
    def to_list(column):
        return eval(column)

    df.rename(columns={'tokens': 'texto'}, inplace=True)

    df['texto'] = df['texto'].apply(to_list)

    # Selecciona las columnas deseadas del DataFrame unlabeled_df
    unlabeled_df_selected = unlabeled_df[['id', 'predict', 'json_comprobacion']]

    # Selecciona las columnas deseadas del DataFrame df
    df_selected = df[['id', 'texto']]

    # Realiza un merge horizontal utilizando la columna id como clave
    merged_df = pd.merge(unlabeled_df_selected, df_selected, on='id', how='inner')

    # Reorganiza el orden de las columnas
    merged_df = merged_df.reindex(columns=['id', 'texto', 'predict', 'json', 'json_comprobacion'])



    # Diccionario de plantilla con las claves especificadas
    plantilla_reglas = {
        'nombre_cliente': [],
        'dni_cliente': [],
        'calle_cliente': [],
        'cp_cliente': [],
        'población_cliente': [],
        'provincia_cliente': [],
        'nombre_comercializadora': [],
        'cif_comercializadora': [],
        'dirección_comercializadora': [],
        'cp_comercializadora': [],
        'población_comercializadora': [],
        'provincia_comercializadora': [],
        'número_factura': [],
        'inicio_periodo': [],
        'fin_periodo': [],
        'importe_factura': [],
        'fecha_cargo': [],
        'consumo_periodo': [],
        'potencia_contratada': []
    }

    # Crear una nueva columna 'reglas' con el diccionario de plantilla
    merged_df['reglas'] = merged_df.apply(lambda x: plantilla_reglas.copy(), axis=1)

    # Expresiones regulares para buscar DNIs/NIEs, CIFs, el importe y el consumo
    dni_regex = r'\b(?:\d{8}[A-Z]|\d{8}-[A-Z]|[XYZ]\d{7}[A-Z]|[XYZ]-\d{7}-[A-Z]|[XYZ]\d{7}-[A-Z])\b'
    cif_regex = r'\b(?:[A-HJUV]\d{8}|[A-HJUV]\d{8}-)\b'
    importe_regex = r'\b(\d+[.,]?\d*)\s*(€|euros)\b'
    consumo_regex = r'\b(\d+[.,]?\d*)\s*(kWh|kilovatio hora|Kilovatio hora|KILOVATIO HORA)\b'
    potencia_regex = r'\b\d+(?:[,.]\d+)?\s*(?:kW|kilovatio|Kilovatio|KILOVATIO)\b'

    def buscar_dni_nie(texto_lista):
        """Busca DNIs y NIEs en una lista de texto y devuelve el más común."""
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(dni_regex, texto)
        if coincidencias:
            contador = Counter(coincidencias)
            # Devuelve el más común, en caso de empate, el primero encontrado
            return contador.most_common(1)[0][0]
        return None

    def buscar_cif(texto_lista):
        """Busca CIFs en una lista de texto y devuelve el más común."""
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(cif_regex, texto)
        if coincidencias:
            contador = Counter(coincidencias)
            # Devuelve el más común, en caso de empate, el primero encontrado
            return contador.most_common(1)[0][0]
        return None

    def buscar_importe(texto_lista):
        """Busca importes en una lista de texto y devuelve el más alto."""
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(importe_regex, texto)
        if coincidencias:
            # Convertir todas las coincidencias a números flotantes
            importes = [float(importe.replace(',', '.')) for importe, _ in coincidencias]
            # Verificar si el importe más alto es 0 euros
            if max(importes) == 0:
                return 0
            # Devolver el valor máximo
            return max(importes)
        return None  # Devolver None si no se encuentra ninguna coincidencia

    def buscar_consumo(texto_lista):
        """Busca consumos en una lista de texto y devuelve la diferencia entre las dos mayores coincidencias."""
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(consumo_regex, texto)
        if coincidencias:
            # Convertir todas las coincidencias a números flotantes
            consumos = [float(consumo.replace(',', '.')) for consumo, _ in coincidencias]
            # Tomar las dos mayores coincidencias
            dos_mayores = nlargest(2, consumos)
            # Calcular la diferencia entre las dos mayores coincidencias
            diferencia = dos_mayores[0] - dos_mayores[1]
            return diferencia if diferencia != 0 else 0  # Devolver 0 si la diferencia es 0
        return None

    def buscar_potencia_contratada(texto_lista):
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(potencia_regex, texto)
        if coincidencias:
            # Ordenar las coincidencias por longitud de cadena de mayor a menor
            coincidencias_ordenadas = sorted(coincidencias, key=len, reverse=True)
            # Tomar la coincidencia más larga
            potencia_contratada = coincidencias_ordenadas[0]
            # Si la potencia contratada es 0, devolver 0
            if potencia_contratada == '0':
                return 0
            return potencia_contratada
        return None

    def actualizar_reglas(row):
        """Actualiza la columna 'reglas' con el DNI o NIE, CIF, el importe más alto, la diferencia entre los dos mayores consumos y la potencia contratada más larga encontrada en 'texto'."""
        texto_lista = row['texto']
        dni_nie_mas_comun = buscar_dni_nie(texto_lista)
        cif_mas_comun = buscar_cif(texto_lista)
        importe_mas_alto = buscar_importe(texto_lista)
        diferencia_consumo = buscar_consumo(texto_lista)
        potencia_contratada = buscar_potencia_contratada(texto_lista)

        if dni_nie_mas_comun:
            row['reglas']['dni_cliente'] = [dni_nie_mas_comun]
        if cif_mas_comun:
            row['reglas']['cif_comercializadora'] = [cif_mas_comun]

        if importe_mas_alto is not None and importe_mas_alto > 0:
            row['reglas']['importe_factura'] = [importe_mas_alto]
        else:
            row['reglas']['importe_factura'] = [0]  # Si es 0, establecer el importe como 0

        if diferencia_consumo is not None:  # Verificar si la diferencia es None o 0
            row['reglas']['consumo_periodo'] = [diferencia_consumo]
        else:
            row['reglas']['consumo_periodo'] = [0]  # Si es None, establecer la diferencia como 0
        if potencia_contratada is not None:  # Verificar si se encontró alguna potencia contratada
            row['reglas']['potencia_contratada'] = [potencia_contratada]
        else:
            row['reglas']['potencia_contratada'] = [0]  # Si no se encuentra ninguna, establecer la potencia como 0

        return row

    # Aplicar la función a cada fila del DataFrame
    merged_df = merged_df.apply(actualizar_reglas, axis=1)

    # Expresión regular para buscar fechas en formato DD.MM.YYYY
    fecha_regex = r'\b\d{2}\.\d{2}\.\d{4}\b'

    def buscar_fechas(texto_lista):
        """Busca fechas en una lista de texto y devuelve una lista de objetos datetime."""
        texto = " ".join(texto_lista)  # Convertir la lista de strings en una sola cadena
        coincidencias = re.findall(fecha_regex, texto)
        fechas = []
        for coincidencia in coincidencias:
            try:
                fecha = datetime.strptime(coincidencia, '%d.%m.%Y')
                fechas.append(fecha)
            except ValueError:
                continue
        return fechas

    def actualizar_fechas(row):
        """Actualiza las claves de fechas en la columna 'reglas' usando las fechas más antiguas encontradas en 'texto'."""
        texto_lista = row['texto']
        fechas = buscar_fechas(texto_lista)
        fechas_unicas = sorted(set(fechas))

        # Asignar las fechas a las claves correspondientes en 'reglas'
        if len(fechas_unicas) > 0:
            row['reglas']['inicio_periodo'] = [fechas_unicas[0].strftime('%d.%m.%Y')]
        if len(fechas_unicas) > 1:
            row['reglas']['fin_periodo'] = [fechas_unicas[1].strftime('%d.%m.%Y')]
        if len(fechas_unicas) > 2:
            row['reglas']['fecha_cargo'] = [fechas_unicas[2].strftime('%d.%m.%Y')]

        return row

    # Aplicar la función a cada fila del DataFrame
    merged_df = merged_df.apply(actualizar_fechas, axis=1)

    def actualizar_json_comprobacion(row):
        json_comprobacion = row['json_comprobacion']
        reglas = row['reglas']

        for key, value in reglas.items():
            if key in json_comprobacion and not json_comprobacion[key]:
                json_comprobacion[key] = value

        return json_comprobacion

    merged_df['json_comprobacion'] = merged_df.apply(actualizar_json_comprobacion, axis=1)



    # Cargar el DataFrame
    csv_path = "./Data/codigos_postales_municipios.csv"
    df_cp = pd.read_csv(csv_path)

    # Lista de provincias con sus códigos
    provincias = {
        1: 'Álava',
        2: 'Albacete',
        3: 'Alicante',
        4: 'Almería',
        5: 'Ávila',
        6: 'Badajoz',
        7: 'Baleares',
        8: 'Barcelona',
        9: 'Burgos',
        10: 'Cáceres',
        11: 'Cádiz',
        12: 'Castellón',
        13: 'Ciudad Real',
        14: 'Córdoba',
        15: 'A Coruña',
        16: 'Cuenca',
        17: 'Girona',
        18: 'Granada',
        19: 'Guadalajara',
        20: 'Guipúzcoa',
        21: 'Huelva',
        22: 'Huesca',
        23: 'Jaén',
        24: 'León',
        25: 'Lleida',
        26: 'La Rioja',
        27: 'Lugo',
        28: 'Madrid',
        29: 'Málaga',
        30: 'Murcia',
        31: 'Navarra',
        32: 'Ourense',
        33: 'Asturias',
        34: 'Palencia',
        35: 'Las Palmas',
        36: 'Pontevedra',
        37: 'Salamanca',
        38: 'Santa Cruz de Tenerife',
        39: 'Cantabria',
        40: 'Segovia',
        41: 'Sevilla',
        42: 'Soria',
        43: 'Tarragona',
        44: 'Teruel',
        45: 'Toledo',
        46: 'Valencia',
        47: 'Valladolid',
        48: 'Vizcaya',
        49: 'Zamora',
        50: 'Zaragoza',
        51: 'Ceuta',
        52: 'Melilla'
    }

    # Convertir la columna "codigo_postal" a tipo string
    df_cp['codigo_postal'] = df_cp['codigo_postal'].astype(str)

    # Añadir ceros por la izquierda para que todos los códigos postales tengan cinco dígitos
    df_cp['codigo_postal'] = df_cp['codigo_postal'].apply(lambda x: x.zfill(5))

    # Crear la columna "provincia" y asignar el nombre de la provincia correspondiente a cada código postal
    df_cp['provincia'] = df_cp['codigo_postal'].str[:2].astype(int).map(provincias)

    # Función para transformar nombres de municipios
    def transformar_nombre_municipio(nombre):
        if ',' in nombre:
            partes = nombre.split(', ')
            if len(partes) == 2:
                return f"{partes[1]} {partes[0]}"
        return nombre

    # Aplicar la función a la columna 'municipio_nombre'
    df_cp['municipio_nombre'] = df_cp['municipio_nombre'].apply(transformar_nombre_municipio)

    def completar_informacion_cliente(row):
        json_comprobacion = row['json_comprobacion']

            # Verificar si 'población_cliente' está presente y está vacía
        if 'población_cliente' in json_comprobacion and not json_comprobacion['población_cliente']:
            if 'cp_cliente' in json_comprobacion and json_comprobacion['cp_cliente']:
                codigo_postal = json_comprobacion['cp_cliente'][0]
                municipio_info = df_cp[df_cp['codigo_postal'] == codigo_postal]
                if not municipio_info.empty:
                    municipio = municipio_info['municipio_nombre'].iloc[0]
                    json_comprobacion['población_cliente'] = [municipio]

        # Verificar si 'provincia_cliente' está presente y está vacía
        if 'provincia_cliente' in json_comprobacion and not json_comprobacion['provincia_cliente']:
            if 'cp_cliente' in json_comprobacion and json_comprobacion['cp_cliente']:
                codigo_postal = json_comprobacion['cp_cliente'][0]
                provincia_info = df_cp[df_cp['codigo_postal'] == codigo_postal]
                if not provincia_info.empty:
                    provincia = provincia_info['provincia'].iloc[0]
                    json_comprobacion['provincia_cliente'] = [provincia]

        # Verificar si 'población_comercializadora' está presente y está vacía
        if 'población_comercializadora' in json_comprobacion and not json_comprobacion['población_comercializadora']:
            if 'cp_comercializadora' in json_comprobacion and json_comprobacion['cp_comercializadora']:
                codigo_postal_com = json_comprobacion['cp_comercializadora'][0]
                municipio_info_com = df_cp[df_cp['codigo_postal'] == codigo_postal_com]
                if not municipio_info_com.empty:
                    municipio_com = municipio_info_com['municipio_nombre'].iloc[0]
                    json_comprobacion['población_comercializadora'] = [municipio_com]

        # Verificar si 'provincia_comercializadora' está presente y está vacía
        if 'provincia_comercializadora' in json_comprobacion and not json_comprobacion['provincia_comercializadora']:
            if 'cp_comercializadora' in json_comprobacion and json_comprobacion['cp_comercializadora']:
                codigo_postal_com = json_comprobacion['cp_comercializadora'][0]
                provincia_info_com = df_cp[df_cp['codigo_postal'] == codigo_postal_com]
                if not provincia_info_com.empty:
                    provincia_com = provincia_info_com['provincia'].iloc[0]
                    json_comprobacion['provincia_comercializadora'] = [provincia_com]

        return json_comprobacion

    # Aplicar la función a cada fila del DataFrame merged_df
    merged_df['json_comprobacion'] = merged_df.apply(completar_informacion_cliente, axis=1)

    # Guardar los resultados en un nuevo archivo CSV
    output_path = "./Data/dataset_resultados.csv"
    merged_df.to_csv(output_path, index=False)


    csv_path = "./Data/dataset_resultados.csv"
    df = pd.read_csv(csv_path)

    # Función para convertir JSON con comillas simples a diccionario
    def convertir_json_a_diccionario(json_str):
        try:
            return ast.literal_eval(json_str)
        except (ValueError, SyntaxError) as e:
            print(f"Error al convertir JSON: {e}")
            return None

    # Aplicar la función a la columna 'json'
    #df['json'] = df['json'].apply(convertir_json_a_diccionario)
    df['reglas'] = df['reglas'].apply(convertir_json_a_diccionario)


    # Función para convertir la cadena de lista a una lista real
    def convertir_a_lista(cadena):
        try:
            lista = ast.literal_eval(cadena)
            if isinstance(lista, list):
                return lista
        except (SyntaxError, ValueError):
            pass  # En caso de error, simplemente retorna una lista vacía
        return []


    df['texto'] = df['texto'].apply(convertir_a_lista)

    # Función para extraer el valor numérico de un string y mantener coma como separador decimal
    def extraer_valor_numerico(value):
        if isinstance(value, (int, float)):
            return f"{value}".replace('.', ',')
        elif isinstance(value, str):
            match = re.search(r'[-+]?\d*[.,]?\d+', value)
            if match:
                return match.group().replace('.', ',')
        return value


    # Función para transformar cada fila
    def transformar_json_comprobacion(json_str):
        # Convertir el string del diccionario a un diccionario real
        diccionario = ast.literal_eval(json_str)

        # Procesar las claves específicas primero
        claves_especificas = ['importe_factura', 'consumo_periodo', 'potencia_contratada']
        for clave in claves_especificas:
            if clave in diccionario and diccionario[clave]:
                diccionario[clave][0] = extraer_valor_numerico(diccionario[clave][0])


        # Iterar sobre los items del diccionario
        for key, value in diccionario.items():
            if key == 'consumo_periodo':
                diccionario[key] = int(float(value[0].replace(',', '.'))) if value else 0
            else:
                diccionario[key] = str(value[0]) if isinstance(value, list) and value else ""

        return diccionario


    # Aplicar la función a cada fila del DataFrame
    df['resultados'] = df['json_comprobacion'].apply(transformar_json_comprobacion)

    # Función para modificar los valores en los diccionarios
    def modificar_direcciones(diccionario):
        if 'calle_cliente' in diccionario:
            diccionario['calle_cliente'] = diccionario['calle_cliente'].replace('S/ N', 'S/N')
        if 'dirección_comercializadora' in diccionario:
            diccionario['dirección_comercializadora'] = diccionario['dirección_comercializadora'].replace('S/ N', 'S/N')
        return diccionario

    # Aplicar la función a la columna 'resultados'
    df['resultados'] = df['resultados'].apply(modificar_direcciones)

    # Función para eliminar espacios innecesarios en nombre_comercializadora
    def modificar_nombre_comercializadora(diccionario):
        if 'nombre_comercializadora' in diccionario:
            diccionario['nombre_comercializadora'] = re.sub(r'\s*([./-])\s*', r'\1', diccionario['nombre_comercializadora'])
        return diccionario

    # Aplicar las funciones a la columna 'resultados'
    df['resultados'] = df['resultados'].apply(modificar_nombre_comercializadora)

    df_resultados = df[['id', 'resultados']]

    # Expander la columna 'resultados' en múltiples columnas
    df_expanded = df_resultados['resultados'].apply(pd.Series)

    # Concatenar las nuevas columnas con el DataFrame original
    df_final = pd.concat([df_resultados.drop(columns=['resultados']), df_expanded], axis=1)

    # Preguntar al usuario si desea sustituir id por número_factura
    sustituir = st.radio("¿Desea substituir 'id' por 'número_factura'?", ("No", "Sí"))

    # Modificar el DataFrame según la elección del usuario
    if sustituir == "Sí":
        df_final = df_final.drop(columns=['id'])
        cols = ['número_factura'] + [col for col in df_final.columns if col != 'número_factura']
        df_final = df_final[cols]

    df_final = df_final.rename(columns=lambda x: x.replace('_', ' '))

    st.dataframe(df_final)


    # Desplegable para seleccionar el formato de descarga
    opcion_descarga = st.selectbox(
        "Seleccione el formato de descarga",
        ("Seleccionar", "Excel", "CSV")
    )


    # Función para convertir DataFrame a CSV
    def convertir_a_csv(df):
        return df.to_csv(index=False).encode('utf-8')


    # Función para convertir DataFrame a Excel
    def convertir_a_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data

    # Opción de descarga
    if opcion_descarga == "Excel":
        st.download_button(
            label="Descargar Excel",
            data=convertir_a_excel(df_final),
            file_name="datos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    elif opcion_descarga == "CSV":
        st.download_button(
            label="Descargar CSV",
            data=convertir_a_csv(df_final),
            file_name="datos.csv",
            mime="text/csv"
        )





