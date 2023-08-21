---
hide-toc: true
firstpage:
lastpage:
---

<!-- ```{toctree} -->
<!-- :hidden: -->
<!-- :caption: Introduction -->
<!-- content/basic_usage -->
<!-- content/dataset_standards -->
<!-- content/minari_cli -->
<!-- ``` -->

<!-- ```{toctree} -->
<!-- :hidden: -->
<!-- :caption: API -->
<!-- api/minari_functions -->
<!-- api/minari_dataset -->
<!-- api/data_collector -->
<!-- ``` -->

<!-- ```{toctree} -->
<!-- :hidden: -->
<!-- :glob: -->
<!-- :caption: Tutorials -->
<!-- tutorials/**/index -->
<!-- ``` -->

<!-- ```{toctree} -->
<!-- :hidden: -->
<!-- :caption: Datasets -->
<!-- datasets/door -->
<!-- datasets/hammer -->
<!-- datasets/relocate -->
<!-- datasets/pen -->
<!-- datasets/pointmaze -->
<!-- datasets/kitchen -->
<!-- ``` -->

<!-- ```{toctree} -->
<!-- :hidden: -->
<!-- :caption: Development -->

<!-- Github <https://github.com/sakakibara-yuuki/assistant> -->
<!-- release_notes/index -->
<!-- ``` -->

<!-- ```{project-logo} _static/img/minari-text.png -->
<!-- :alt: Minari Logo -->
<!-- ``` -->

# Assistant For You !
build bookshelf, and chat! your assistant !


<div id="termynal" data-termynal>
  <span data-ty="input">git clone https://github.com/sakakibara-yuuki/assistant.git</span>
  <span data-ty="input">pip install -e .</span>
  <span data-ty="progress"></span>
  <span data-ty>Successfully installed assistant</span>
  <span data-ty></span>
  <span data-ty="input">python -m assistant bookshelf -u reference.yaml</span>
  <span data-ty>Update Bookshelf</span>
  <span data-ty></span>
  <span data-ty="input">python -m assistant chat</span>
  <span data-ty="input" data-ty-prompt="you :">what is CQL?</span>
  <span data-ty="input" data-ty-prompt="A   :">CQL is 'Conservative Q-Learning' that is ...</span>
  <span data-ty="input" data-ty-prompt="you :">Bye</span>
  <span data-ty="input" data-ty-prompt="A   :">bye!</span>
  <span data-ty></span>
  <span data-ty="input">python -m assistant qa</span>
  <span data-ty="input" data-ty-prompt="you :">what is CQL?</span>
  <span data-ty="input" data-ty-prompt="A   :">CQL is 'Conservative Q-Learning' that is ...</span>
</div>

## requirements

- python >= 3.11


## installation

```sh
git clone https://github.com/sakakibara-yuuki/assistant.git
pip install -e .
```

## How to Use

**create bookshelf**

```sh
python -m assistant bookshelf -u reference.yaml
```

**chat assistant**

```sh
python -m assistant chat
```
if you end, input `bye`

**QA assistant**

```sh
python -m assistant qa
```
