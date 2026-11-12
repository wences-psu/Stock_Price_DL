""" Function to display source code in a collapsible format """

from IPython.display import Markdown, display
import inspect

def show_source(func):
    """Display a collapsible, syntax-highlighted view of a function's source code."""
    src = inspect.getsource(func)
    md = f"""
<details>
<summary>View implementation of <code>{func.__name__}</code></summary>

```python
{src}
```

</details>
"""
    display(Markdown(md))
