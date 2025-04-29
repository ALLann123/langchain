#!/usr/bin/python3
import streamlit as st
from streamlit_option_menu import option_menu
import about, account, home, trending, your_post

#this sets the page title
st.set_page_config(
    page_title="Law Agent"
)

class MultiApp:
    #class constructor that when an object is created executes and creates an empty list called apps
    def __init__(self):
        self.apps=[]
    
    #now lets add our apps into the list using append keyword
    def add_app(self, title, function):
        self.apps.append({
            "title": title,
            "function":function
        })

    def run():
        # this is the syntax for creating a sidebar
        with st.sidebar:
            app = option_menu(
                menu_title="Pondering",
                options=['Home', 'Account', 'Trending', 'Your Post', 'About'],
                icons=['house-fill','person-circle','trophy-fill','chat-fill','info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=1,
                styles={  # corrected from 'style' to 'styles' based on your version
                    "container": {"padding": "5!important", "background-color": "black"},
                    "icon": {"color": "white", "font-size": "23px"},
                    "nav-link": {
                        "color": "white",
                        "font-size": "20px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "blue",
                    },
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )
            st.title("Upload File Here")
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv", "jpg", "png", "pdf"])

        if app=='Home':
            #this will run the home page
            home.app()

        if app=='Account':
            account.app(uploaded_file)
        
        if app=='Trending':
            trending.app()
        
        if app=='Your Post':
            your_post.app(uploaded_file)
        
        if app=='about':
            about.app()

    run()



