## LocalTemplate

### 2021/09/16 update - We are currently cleaning the codes, a cleaned version of local tempalte extractor will be updated once finished.

LocalTemplate is a python tool to extract the general chemical reaction template from a reaction.
The output of the template extractor includes both local reaction template and reaction sites (atom index and atom map number).
* **template_extractor.py** is modified from [RDChiral](https://github.com/connorcoley/rdchiral) to extract more general reaction template.<br>
* **template_extract_utils.py** has the supporting scripts to help sorting the template and parsing the correct edit sites.<br>
* **template_decoder.py** is the script used to apply the predicted template to product(s) to make reactant(s).<br>
* **template_includes_ring.py** is designed for those special case where the predicted edit site is in a ring.<br>