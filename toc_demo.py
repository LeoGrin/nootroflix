import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(
    page_title=" Demo App",
    page_icon="🧊",  # EP: how did they find a symbol?
    layout="wide",
    initial_sidebar_state="expanded",
)


def md_link(text: str, url: str):
    return f"[{text}]({url})"


def api_docs(anchor):
    url = f"https://docs.streamlit.io/en/stable/api.html#{anchor}"
    text = f"docs - {anchor}"
    st.markdown("---")
    st.markdown(md_link(text, url))


st.sidebar.title("This is sidebar")

st.title("Let's try Streamlit")
st.markdown(
    "[Streamlit](https://www.streamlit.io/) is a great way to make a dashboard, "
    "interactive or not."
)

col_a, col_b = st.beta_columns(2)

with col_a:
    st.subheader("Great features")
    st.markdown(
        """
    - you just write a python script that will render as webpage
    - there is a sleek HTML template, no extra formatting needed 
    - interactivity - you can get feedback from widgets and controls
    - native methods to draw charts, render latex, insert graphviz
    - dedicated hosting (in beta)
"""
    )

with col_b:
    st.subheader("You can see Streamlit as")
    st.markdown(
        """
- [Jupyter notebook](https://jupyter.org/) less code cells
- beefed-up [Handout][handout] (or Julia [Pluto.jl][pluto])
- [Flask][flask] without template setup
- Rstudio's [Shiny][shiny] in Python, with hosting
- another way to use [Plotly Dash](https://plotly.com/dash/) and [Bokeh](https://docs.bokeh.org/en/latest/index.html)
- the 'iPhone of Python'
[shiny]: https://shiny.rstudio.com/
[pluto]: https://github.com/fonsp/Pluto.jl
[handout]: https://github.com/danijar/handout    
[flask]: https://flask.palletsprojects.com/en/1.1.x/
    """
    )

st.header("Minimal example")

"""
1. Install: 
```
$ pip install streamlit
```
After installation ```streamlit``` will be available as a command line tool and as a package.
2. Make `my_app.py`:
```python
import streamlit as st
st.write("Hello, world!")
```
3. Run:
```
$ streamlit run my_app.py
```
4. Point your broswer to http://localhost:8501. The page will refresh as you edit and save `my_app.py`.
5. Learn more with [tutorials](https://docs.streamlit.io/en/stable/getting_started.html).
"""

st.header("Small examples")

st.subheader("Input and display a number, show code")

with st.echo():
    x = st.number_input("A number please:")
    st.write("Just got", x)

st.subheader("Input and display a number, show code")

color = st.select_slider(
    "Select a color of the rainbow",
    options=["red", "orange", "yellow", "green", "blue", "indigo", "violet"],
)
st.write("My favorite color is", color)


st.subheader("Slider")

hour = st.slider("Hour", 0, 23, 12)

st.subheader("Display text, markdown, latex, variable, code")

st.write("<hr>")
st.text("Fixed width text")
st.markdown("_Markdown_")  # see *
st.latex(r"e^{i\pi} + 1 = 0")
st.write("Most objects")  # df, err, func, keras!
st.write(dict(a=1))
st.write(["st", "is <", 3])  # see *
st.title("My title")
st.header("My header")
st.subheader("My sub")
st.code("for i in range(8): foo()")

st.subheader("Line break")

st.markdown("---")

st.subheader("Graphviz chart")

st.graphviz_chart(
    """
    digraph {
        run -> intr
        intr -> runbl
        runbl -> run
        run -> kernel
        kernel -> zombie
        kernel -> sleep
        kernel -> runmem
        sleep -> swap
        swap -> runswap
        runswap -> new
        runswap -> runmem
        new -> runmem
        sleep -> runmem
    }
"""
)

st.subheader("Checkbox as collapse control")

if st.checkbox("Show raw data"):
    st.subheader("Raw data")

st.subheader("Input text")

with st.echo():
    name = st.text_input("Name")
    st.text(name)

st.subheader("Show dataframe or table")
st.write(
    pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)

st.subheader("Now there is a chart")
chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.line_chart(chart_data)

st.subheader("Now there is a map")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=["lat", "lon"]
)
st.map(map_data)


st.header("Notes and discussion")

""" 
#### Limitations and feature requests
- cannot add raw html directly (safety concern)
- no obvious way to persist the result as html 
- table of contents (see below)
#### Mind model
- can persist data with  ```@st.cache()```
- show code via ```with st.echo():``` decorator
- multiple selection from list 
"""

st.title("Making TOC (experimental)")
st.markdown(
    """
Related links:
- https://discuss.streamlit.io/t/table-of-contents-widget/3470/8?u=epogrebnyak
- https://github.com/streamlit/streamlit/issues/726
"""
)


class Header:
    tag: str = ""

    def __init__(self, text: str):
        self.text = text

    @property
    def id(self):
        """Create an identifcator from text."""
        return "".join(filter(str.isalnum, self.text)).lower()

    @property
    def anchor(self):
        """Provide html text for anchored header. Example:
           <h1 id="abcdef">Abc Def</h1>
        """
        return f"<{self.tag} id='{self.id}'>{self.text}</{self.tag}>"

    def toc_item(self) -> str:
        """Make markdown item for TOC listing. Example:
           '  - <a href='#abc'>Abc</a>'
        """
        return f"{self.spaces}- [{self.text}]('#{self.id}')"

    @property
    def spaces(self):
        return dict(h1="", h2=" " * 2, h3=" " * 4).get(self.tag)


assert Header("abc").spaces is None


class H1(Header):
    tag = "h1"


class H2(Header):
    tag = "h2"


assert H2("Abc").toc_item() == "  - [Abc]('#abc')"


class H3(Header):
    tag = "h3"


class TOC:
    """
    Original code, used with modifications:
    https://discuss.streamlit.io/t/table-of-contents-widget/3470/8?u=epogrebnyak
    """

    def __init__(self):
        self._headers = []
        self._placeholder = st.empty()

    def title(self, text):
        self._add(H1(text))

    def header(self, text):
        self._add(H2(text))

    def subheader(self, text):
        self._add(H3(text))

    def generate(self):
        text = "\n".join([h.toc_item() for h in self._headers])
        self._placeholder.markdown(text, unsafe_allow_html=True)

    def _add(self, header):
        st.markdown(header.anchor, unsafe_allow_html=True)
        self._headers.append(header)


class TOC_Sidebar(TOC):
    def __init__(self):
        self._headers = []
        self._placeholder = st.sidebar.empty()


def blah():
    for a in range(3):
        st.write("Blabla...")


toc = TOC()

toc.title("Title")

toc.header("Header 1")
blah()

toc.header("Header 2")
blah()

toc.subheader("Subheader 2.1")
blah()

toc.subheader("Subheader 2.2")
blah()

toc.generate()