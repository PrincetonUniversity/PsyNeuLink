# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# **************************************  TechnicalNote *************************************************

"""
This is a Sphinx extension that enables a custom RST directive. It is based on the standard docutils Container
directive, though it does not inherit from that directive as there are key changes that made subclassing unsuitable.

The purpose of this directive is to wrap arbitrary content in a technical-note html class, thus allowing for it to be
hidden or shown by way of embedded javascript in the doc.

"""

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from sphinx.util.nodes import nested_parse_with_titles

class TechnicalNote(Directive):
    # optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {'name': directives.unchanged}
    has_content = True

    def run(self):
        self.assert_has_content()
        text = '\n'.join(self.content)
        try:
            if self.arguments:
                classes = directives.class_option(self.arguments[0])
            else:
                classes = []
            classes.append('technical-note')
        except ValueError:
            raise self.error(
                'Invalid class attribute value for "%s" directive: "%s".'
                % (self.name, self.arguments[0]))
        node = nodes.container(text)
        node['classes'].extend(classes)
        nested_parse_with_titles(self.state, self.content, node)
        return [node]

def setup(app):
    app.add_directive("technical_note", TechnicalNote)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
