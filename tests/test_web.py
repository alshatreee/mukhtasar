"""Tests for web.py — HTML cleaning and URL fetching."""

from mukhtasar.web import clean_html


class TestCleanHtml:
    def test_strips_script_tags(self):
        html = "<p>نص مهم</p><script>alert('x')</script><p>نص آخر</p>"
        result = clean_html(html)
        assert "alert" not in result
        assert "نص مهم" in result

    def test_strips_style_tags(self):
        html = "<style>.red{color:red}</style><p>المحتوى هنا مهم جداً</p>"
        result = clean_html(html)
        assert "color" not in result
        assert "المحتوى" in result

    def test_strips_nav_footer(self):
        html = "<nav>قائمة التنقل</nav><p>المحتوى الرئيسي هنا</p><footer>حقوق النشر</footer>"
        result = clean_html(html)
        assert "قائمة التنقل" not in result
        assert "المحتوى الرئيسي" in result

    def test_converts_br_to_newline(self):
        html = "سطر أول<br>سطر ثاني"
        result = clean_html(html)
        assert "\n" in result

    def test_converts_p_to_newline(self):
        html = "<p>فقرة أولى</p><p>فقرة ثانية</p>"
        result = clean_html(html)
        assert "\n" in result

    def test_decodes_html_entities(self):
        html = "أحمد &amp; محمد &lt;أصدقاء&gt;"
        result = clean_html(html)
        assert "&" in result
        assert "<" in result

    def test_decodes_numeric_entities(self):
        # &#1575; = ا (Arabic Alef)
        html = "&#1575;&#1604;&#1587;&#1604;&#1575;&#1605;"
        result = clean_html(html)
        assert "السلام" in result

    def test_removes_excess_whitespace(self):
        html = "كلمة    أولى     كلمة    ثانية"
        result = clean_html(html)
        assert "    " not in result

    def test_empty_html(self):
        assert clean_html("") == ""

    def test_plain_text_passes_through(self):
        text = "هذا نص عادي بدون HTML"
        assert clean_html(text) == text

    def test_nested_tags(self):
        html = "<div><p><strong>نص مهم</strong> جداً</p></div>"
        result = clean_html(html)
        assert "نص مهم" in result
        assert "جداً" in result
        assert "<" not in result
