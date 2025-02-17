import io
import joblib
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Cargar los modelos previamente entrenados
def load_models():
    model_digit = joblib.load("models/output/KNN_Digitos_v2.joblib")
    model_operator = joblib.load("models/output/SVM_Operadores.joblib")
    return model_digit, model_operator

# Preprocesar la imagen para que sea compatible con el modelo MNIST
def transform_image_to_mnist(image_data):
    # Convertir la imagen en un array de píxeles
    img = Image.fromarray(image_data.astype(np.uint8))
    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    st.image(img, caption="Prueba de imagen")

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

def play_canvas1():
    st.set_page_config(layout = "wide")
    #Cargar los modelos
    #d_model, op_model = load_models()  DESCOMENTAR ESTO

    # Creando variables del sidebar
    stroke_color = "white"
    bg_color = "black"
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    with st.container():
        (
            number_one,
            _,
            operator_one,
            number_two,
            _,
            operator_two,
            number_three,
        ) = st.columns([3, 1, 2, 3, 1, 2, 3])

        with number_one:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=4,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=35,
                    width=35,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_1",
                )

            number_1 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=15,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=140,
                width=140,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_1",
            )

        with operator_one:
            with st.container():
                st.markdown("#")
                st.markdown("#")
                operator_1 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=8,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=70,
                    width=70,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="operator_1",
                )
        with number_two:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_2 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=4,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=35,
                    width=35,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_2",
                )
            number_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=15,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=140,
                width=140,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_2",
            )

        with operator_two:
            st.markdown("#")
            st.markdown("#")
            operator_2 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=8,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=70,
                width=70,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="operator_2",
            )

        with number_three:
            c1, c2 = st.columns(2)
            with c1:
                st.empty()
            with c2:
                exponent_3 = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=4,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image=None,
                    update_streamlit=realtime_update,
                    height=35,
                    width=35,
                    drawing_mode="freedraw",
                    point_display_radius=0,
                    key="exponent_3",
                )

            number_3 = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=15,
                stroke_color=stroke_color,
                background_color=bg_color,
                background_image=None,
                update_streamlit=realtime_update,
                height=140,
                width=140,
                drawing_mode="freedraw",
                point_display_radius=0,
                key="number_3",
            )
    #AÑADIR ACÁ LA LOGICA DE LA PREDICCION
    
    #Primer digito
    if number_1.image_data is not None:
        transform_image_to_mnist(number_1.image_data)
    #Primer exponente
    if exponent_1.image_data is not None:
        transform_image_to_mnist(exponent_1.image_data)

    #Primer operador
    if operator_1.image_data is not None:
        transform_image_to_mnist(operator_1.image_data)
    
    #Segundo digito
    if number_2.image_data is not None:
        transform_image_to_mnist(number_2.image_data)
    #segundo exponente
    if exponent_2.image_data is not None:
        transform_image_to_mnist(exponent_2.image_data)

    #Segundo operador
    if operator_2.image_data is not None:
        transform_image_to_mnist(operator_2.image_data)

    #Tercer digito
    if number_3.image_data is not None:
        transform_image_to_mnist(number_3.image_data)
    #tercer exponente
    if exponent_3.image_data is not None:
        transform_image_to_mnist(exponent_3.image_data)


# Ejecutar la función principal
def main():
    play_canvas1()

if __name__ == "__main__":
    main()
