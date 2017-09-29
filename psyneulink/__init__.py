# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***********************************************  Init ****************************************************************
import logging as _logging


# https://stackoverflow.com/a/17276457/3131666
class _Whitelist(_logging.Filter):
    def __init__(self, *whitelist):
        self.whitelist = [_logging.Filter(name) for name in whitelist]

    def filter(self, record):
        return any(f.filter(record) for f in self.whitelist)


class _Blacklist(_Whitelist):
    def filter(self, record):
        return not _Whitelist.filter(self, record)


_logging.basicConfig(
    level=_logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for handler in _logging.root.handlers:
    handler.addFilter(_Blacklist(
        'psyneulink.scheduling.scheduler',
        'psyneulink.scheduling.condition',
    ))

