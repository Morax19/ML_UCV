import io
import joblib
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

#Ajustes iniciales de la página
st.set_page_config(layout = "wide")

# Preprocesar la imagen para que sea compatible con el modelo
def to_model(image_data):
    # Convertir la imagen en un array de píxeles
    img = Image.fromarray(image_data.astype(np.uint8))
    img = img.convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = img.reshape(img.shape[0], 28 * 28)
    img = img / 255.0
    #st.image(img, caption="Prueba de imagen")
    return img

def get_number(prediction):
    return str(prediction[0])

def get_sign(prediction):
    val = prediction[0]
    if val == ord('+'):
        return '+'
    elif val == ord('-'):
        return '-'
    elif val == ord('*') or val == ord('×'):
        return '*'
    elif val == ord('/') or val == ord('÷'):
        return '/' 


def play_canvas1(d_model, op_model):
    # Creando variables del sidebar
    stroke_color = "white"
    bg_color = "black"
    realtime_update = True

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
                    stroke_width=7,
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
                stroke_width=7,
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
    canvases_full = (
        number_1.image_data is not None and exponent_1.image_data is not None and operator_1.image_data is not None and number_2.image_data is not None and exponent_2.image_data is not None and operator_2.image_data is not None and number_3.image_data is not None and exponent_3.image_data is not None
    )

    if st.button("Make Prediction!", disabled = not canvases_full):
        #Predecir dígitos
        d1_pred = d_model.predict(to_model(number_1.image_data))
        d2_pred = d_model.predict(to_model(number_2.image_data))
        d3_pred = d_model.predict(to_model(number_3.image_data))

        #Predecir exponentes
        exp1_pred = d_model.predict(to_model(exponent_1.image_data))
        exp2_pred = d_model.predict(to_model(exponent_2.image_data))
        exp3_pred = d_model.predict(to_model(exponent_3.image_data))

        #Predecir operadores
        op1_pred = op_model.predict(to_model(operator_1.image_data))
        op2_pred = op_model.predict(to_model(operator_2.image_data))

        #Obtener valores en string
        #Digitos
        d1_res = get_number(d1_pred)
        d2_res = get_number(d2_pred)
        d3_res = get_number(d3_pred)

        #Exponentes
        exp1_res = get_number(exp1_pred)
        exp2_res = get_number(exp2_pred)
        exp3_res = get_number(exp3_pred)

        #Operadores
        op1_res = get_sign(op1_pred)
        op2_res = get_sign(op2_pred)

        final_expression = "("+d1_res+"**"+exp1_res+")"+op1_res+"("+d2_res+"**"+exp2_res+")"+op2_res+"("+d3_res+"**"+exp3_res")"
        res = eval(final_expression)
        
        st.write(f"El resultado de la operación es: {res}")

# Ejecutar la función principal
def main():
    #Cargar los modelos
    try:
        d_model = joblib.load("models/output/KNN_Digitos_v2.joblib")
        op_model = joblib.load("models/output/SVM_Operadores.joblib")
    except Exception as e:
        st.error(f"Error cargando los modelos: {e}")
        st.stop()
    
    #Dibujar los canvases
    play_canvas1(d_model, op_model)

if __name__ == "__main__":
    main()