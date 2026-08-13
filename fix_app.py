import re

file_path = r'C:\Users\ALFATEH\Desktop\Safex_week1_group7_Abdul-hadi\app.py'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the literal \n issue in session state initialization
content = content.replace(
    'if "selected_candidate_for_hero" not in st.session_state:\\n        st.session_state.selected_candidate_for_hero = None',
    'if "selected_candidate_for_hero" not in st.session_state:\n        st.session_state.selected_candidate_for_hero = None'
)

# Fix the hero rendering logic with literal \n
content = content.replace(
    '# Display hero with selected candidate score if available\n    selected_candidate_for_hero = st.session_state.get("selected_candidate_for_hero")\n    if selected_candidate_for_hero:\n        st.markdown(_build_hero_html(selected_candidate_for_hero), unsafe_allow_html=True)\n    else:\n        st.markdown(_build_hero_html(st.session_state.get("analysis")), unsafe_allow_html=True)',
    '# Display hero with selected candidate score if available\n    selected_candidate_for_hero = st.session_state.get("selected_candidate_for_hero")\n    if selected_candidate_for_hero:\n        st.markdown(_build_hero_html(selected_candidate_for_hero), unsafe_allow_html=True)\n    else:\n        st.markdown(_build_hero_html(st.session_state.get("analysis")), unsafe_allow_html=True)'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("File fixed successfully")
