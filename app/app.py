import streamlit as st
import torch
import sys
import os
from streamlit_image_comparison import image_comparison

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import ResNetColorizer
from utils import process_image, postprocess_output

st.set_page_config(
    page_title='Landscape Colorizerrrrr',
    page_icon='🌲',
    layout='wide'
)

st.title('⛰️🌋🌲🌳Landscape🌴🌊⛵☃️Colorizerrrrr❄️🌧️🐳🌕')
st.markdown('*Model: ResNet-18 U-Net (Trained on 4,300 Landscape Images)*')

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNetColorizer().to(device)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    model_path = os.path.join(root_dir, 'outputs', 'models', 'model.pth')
    
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            return model, device
        except RuntimeError as e:
            st.error(f'Error loading model: {e}')
            return None, device
    else:
        st.error(f'Model not found at: {model_path}')
        st.warning('Please ensure model.pth is inside \'outputs/models/\'')
        return None, device

model, device = load_model()

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader('Upload a B&W landscape image', type=['jpg', 'png', 'jpeg'])
    st.info(f'Running on: **{str(device).upper()}**')

# main loop
if uploaded_file is not None and model is not None:
    
    with st.spinner('Colorizing...'):
        resized_bw_img, L_tensor = process_image(uploaded_file, device)
        
        with torch.no_grad():
            ab_output = model(L_tensor)
        
        color_result = postprocess_output(L_tensor, ab_output)

    st.subheader('Results Comparison')

    col1, col2 = st.columns(2)
    with col1:
        image_comparison(
            img1=resized_bw_img,
            img2=color_result,
            label1='Input (B&W)',
            label2='Colorized',
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )
    with col2:
        st.image(color_result, caption='Final Output (512x512)', width='content')
        st.download_button(
            label='Download Colorized Image',
            data=open('temp_result.png', 'wb').read() if os.path.exists('temp_result.png') else 'Error',
            file_name='colorized_landscape.png',
            mime='image/png'
        )

else:
    # default state
    # st.image('https://www.eikojones.com/wp-content/uploads/2023/06/that-wanaka-tree-in-autumn.jpg', caption='Example Landscape', width='stretch')
    st.info('Upload an image to begin.')