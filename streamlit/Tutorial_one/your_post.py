import streamlit as st
import base64

def app(uploaded_file=None):
    st.title("Your Post")

    if uploaded_file:
        st.success(f"Displaying: {uploaded_file.name}")

        if uploaded_file.type.startswith("text"):
            content = uploaded_file.read().decode("utf-8")
            st.text_area("File Content", content, height=300)

        elif uploaded_file.type.startswith("image"):
            st.image(uploaded_file, caption=uploaded_file.name)

        elif uploaded_file.type == "application/pdf":
        # Display PDF using an iframe
            base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

    else:
        st.info("No file uploaded yet.")
