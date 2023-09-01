mkdir -p ~/ .streamlit/

echo "\
[server]\n\
port = $PORT\n\
ecableCORS = false\n\
headless = true\n\
\n\
" > ~/ .streamlit/config.html
