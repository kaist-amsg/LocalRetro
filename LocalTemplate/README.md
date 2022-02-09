## LocalTemplate
LocalTemplate is a python tool to extract the general chemical reaction template from a reaction.
The output of the template extractor includes both local reaction template and reaction sites (atom index and atom map number).
* **template_extractor.py** is modified from [RDChiral](https://github.com/connorcoley/rdchiral).<br>
* **template_extract_utils.py** has the supporting scripts to help sorting the template and parsing the correct edit sites.<br>
* **template_decoder.py** is used to apply the predicted templates to product(s) to make reactant(s).<br>