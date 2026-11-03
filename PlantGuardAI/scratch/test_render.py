import streamlit as st
import textwrap

def render_html(snippet: str):
    cleaned = "\n".join(line.strip() for line in snippet.splitlines() if line.strip())
    return cleaned

test_str = """
    <div class="pg-card">
        <div class="pg-card-header">Header</div>
            <div style="background: red;">
                <span>Nested indented tag</span>
            </div>
    </div>
"""

cleaned = render_html(test_str)
print("Cleaned:")
print(cleaned)
assert not any(line.startswith("    ") for line in cleaned.splitlines()), "Should have no 4-space indentation!"
print("Test passed: No 4-space indentation anywhere.")

