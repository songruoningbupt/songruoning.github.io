### 如何在Github上写公式

ZT: http://cwiki.apachecn.org/pages/viewpage.action?pageId=8159393

- 使用MathJax引擎
    - 在Markdown中添加MathJax引擎。`<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>`
    - 然后，再使用Tex写公式。
        - $$公式$$表示行间公式，本来Tex中使用\(公式\)表示行内公式，但因为Markdown中\是转义字符，所以在Markdown中输入行内公式使用\\(公式\\)，如下代码：
            - `$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$`
            - `\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)`


测试案例：

### 机器学习-如何在github上写数学公式

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

居中格式: $$xxx$$
$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
靠左格式: \\(xxx\\)
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)

测试
$$\frac{7x+5}{1+y^2}$$
\\(l(x_i) = - \log_2 P(x_i)\\)