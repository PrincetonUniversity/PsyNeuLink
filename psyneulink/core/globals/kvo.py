# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#


# Key Value Observing (KVO):
#
#      Observed object must implement the following:
#
#      - a dictionary (in its __init__ method) of observers with an entry for each attribute to be observed:
#          key should be string identifying attribute name
#          value should be empty list
#          TEMPLATE:
#              self.observers = {<kpAttribute>: []}  # NOTE: entry should be empty here
#
#      - a method that allows other objects to register to observe the observable attributes:
#          TEMPLATE:
#              def add_observer_for_keypath(self, object, keypath):
#                  self.observers[keypath].append(object)
#
#      - a method that sets the value of each attribute to be observed with the following format
#          TEMPLATE:
#              def set_attribute(self, new_value):
#                  old_value = self.<attribute>
#                  self.<attribute> = new_value
#                  if len(self.observers[<kpAttribute>]):
#                      for observer in self.observers[<kpAttribute>]:
#                          observer.observe_value_at_keypath(<kpAttribute>, old_value, new_value)
#
#      Observing object must implement a method that receives notifications of changes in the observed objects:
#          TEMPLATE
#              def observe_value_at_keypath(keypath, old_value, new_value):
#                  [specify actions to be taken for each attribute (keypath) observed]

# ********************************************* KVO ********************************************************************

__all__ = [
    'classProperty', 'observe_value_at_keypath'
]


class classProperty(property):
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

def observe_value_at_keypath(keypath, old_value, new_value):
    print("KVO keypath: {0};  old value: {1};  new value: {2}".format(keypath, old_value, new_value))


# def is_numeric_or_none(x):
#     if not x:
#         return True
#     if isinstance(x, numbers.Number):
#         return True
#     if isinstance(x, (list, np.ndarray)) and all(isinstance(i, numbers.Number) for i in x):
#         return True
#     else:
#         return False
