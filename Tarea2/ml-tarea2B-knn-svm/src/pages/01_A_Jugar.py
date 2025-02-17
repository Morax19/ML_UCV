import io
import joblib
import numpy as np
import streamlit as st
import streamlit_drawable_canvas as st_canvas
from PIL import Image

# Cargar los modelos previamente entrenados
def load_models():
    model_digit = joblib.load("models/output/KNN_Digitos_v2.joblib")
    model_operator = joblib.load("models/output/SVM_Operadores.joblib")
    return model_digit, model_operator

# Preprocesar la imagen para que sea compatible con el modelo MNIST
def transform_image_to_mnist(image_data):
    # Convertir la imagen en un array de píxeles
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('L')  # Convertir a escala de grises
    img = img.resize((28, 28), Image.ANTIALIAS)
    img_array = np.array(img)
    return img_array

# Función para predecir el dígito o el operador
def predict_digit(model, image):
    # Preprocesar la imagen
    r_image = transform_image_to_mnist(image)
    
    # Normalizar la imagen para que coincida con el formato de MNIST
    r_image = r_image.reshape(r_image.shape[0], 28 * 28)
    r_image = image_resized / 255.0
    
    # Hacer la predicción
    prediction = model.predict(r_image)
    
    # Obtener el valor de la predicción
    predicted_class = prediction.argmax()
    return predicted_class

# Función principal para el canvas interactivo
def play_canvas():
    # Cargar los modelos
    #model_number, model_operator = load_models()

    # Crear variables de la barra lateral
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Contenedor para los números y operadores
    number_1 = st.empty()
    number_2 = st.empty()
    operator_1 = st.empty()

    # Definir el lienzo donde los usuarios dibujarán
    number_1.image_data = st_canvas(
        width=280,
        height=280,
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        realtime_update=realtime_update
    )

    # Predicciones y operación
    with number_1:
        if number_1.image_data is not None:
            st.write("Predicción Número 1: ")
            digit_1 = predict_digit(model_number, number_1.image_data)
            st.write(f"Número 1 Predicho: {digit_1}")
        
        # Aquí puedes repetir el proceso para el número 2 y el operador si lo deseas

    with operator_1:
        if operator_1.image_data is not None:
            operator_pred = predict_digit(model_operator, operator_1.image_data)
            st.write(f"Operador Predicho: {operator_pred}")
        
        # Aquí puedes realizar las operaciones entre los números
        if digit_1 is not None and digit_2 is not None and operator_pred is not None:
            if operator_pred == ord('+'):  # Representación de '+'
                result = digit_1 + digit_2
            elif operator_pred == ord('-'):  # Representación de '-'
                result = digit_1 - digit_2
            elif operator_pred == ord('×') or operator_pred == ord('*'):  # Representación de '×'
                result = digit_1 * digit_2
            elif operator_pred == ('÷') or operator_pred == ord('/'):  # Representación de '/'
                result = digit_1 / digit_2 if digit_2 != 0 else "Error"
            st.write(f"Resultado: {result}")

# Ejecutar la función principal
def main():
    play_canvas()

if __name__ == "__main__":
    main()
