# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************** EMComposition show_graph *************************************************

from psyneulink.core.compositions.showgraph import (
    INITIAL_FRAME,
    ShowGraph,
    ShowGraphError,
    _gv_executable_not_found_error_msg,
    get_default_showgraph_dir,
)
from psyneulink.library.compositions.autodiffcomposition import torch_available

if torch_available:
    from psyneulink.library.compositions.pytorchshowgraph import PytorchShowGraph as _EMCompositionShowGraphBase
else:
    _EMCompositionShowGraphBase = ShowGraph


__all__ = ["EMCompositionShowGraph"]


class EMCompositionShowGraph(_EMCompositionShowGraphBase):
    """ShowGraph subclass that adds EMComposition-specific layout constraints."""

    def _generate_output(self,
                         G,
                         enclosing_comp,
                         active_items,
                         show_controller,
                         output_fmt,
                         context
                         ):
        G = super()._generate_output(G,
                                     enclosing_comp,
                                     active_items,
                                     show_controller,
                                     'gv',
                                     context)
        self._add_emcomposition_ordering_constraints(G)

        composition = self.composition

        if output_fmt == 'pdf':
            from graphviz.backend.execute import ExecutableNotFound
            try:
                G.view(composition.name.replace(" ", "-"),
                       cleanup=True,
                       directory=get_default_showgraph_dir().joinpath('PDFS'))
            except ExecutableNotFound as e:
                raise ShowGraphError(_gv_executable_not_found_error_msg) from e
            except Exception as e:
                raise ShowGraphError(f"Problem displaying graph for {composition.name}: {e}") from e

        elif output_fmt == 'gif':
            if composition.active_item_rendered or INITIAL_FRAME in active_items:
                self._generate_gifs(G, active_items, context)

        elif output_fmt == 'jupyter':
            return G

        elif output_fmt == 'gv':
            return G

        elif output_fmt == 'source':
            return G.source

        elif not output_fmt:
            return None

        else:
            raise ShowGraphError(f"Bad arg in call to {composition.name}.show_graph: '{output_fmt}'.")

    def _add_emcomposition_ordering_constraints(self, G):
        """Add invisible ordering constraints for EMComposition's retrieval layers."""

        composition = self.composition

        def get_node_id_in_G_body(node):
            for item in G.body:
                quoted_items = item.split('"')[1::2]
                if ((quoted_items and node.name == quoted_items[0])
                        or (node.name + ' [' in item)):
                    if '->' not in item:
                        if quoted_items:
                            return quoted_items[0]
                        return item.strip().split(' [', 1)[0]

        def get_node_ids(nodes):
            node_ids = []
            for node in nodes:
                if node is None:
                    continue
                node_id = get_node_id_in_G_body(node)
                if node_id is not None:
                    node_ids.append(node_id)
            return node_ids

        def get_node_id(node):
            node_ids = get_node_ids([node])
            return node_ids[0] if node_ids else None

        query_input_node_ids = get_node_ids(composition.query_input_nodes)
        value_input_node_ids = get_node_ids(composition.value_input_nodes)
        input_node_ids = query_input_node_ids + value_input_node_ids
        field_memory_node_ids = get_node_ids(composition.field_memory_nodes)
        combined_scores_node_ids = get_node_ids([composition.combined_scores_node])
        concatenate_queries_node_id = get_node_id(composition.concatenate_queries_node)
        concatenated_memory_node_id = get_node_id(composition.concatenated_memory_node)
        query_retrieved_node_ids = get_node_ids([field.retrieved_node for field in composition.key_fields])
        value_retrieved_node_ids = get_node_ids([field.retrieved_node for field in composition.value_fields])
        retrieved_node_ids = query_retrieved_node_ids + value_retrieved_node_ids

        def add_same_rank(node_ids):
            if len(node_ids) > 1:
                with G.subgraph() as rank_subgraph:
                    rank_subgraph.attr(rank='same')
                    for node_id in node_ids:
                        rank_subgraph.node(node_id)

        for node_ids in (input_node_ids, field_memory_node_ids, combined_scores_node_ids, retrieved_node_ids):
            add_same_rank(node_ids)

        combined_scores_node_id = combined_scores_node_ids[0] if combined_scores_node_ids else None
        invisible_edge_attrs = {
            'arrowhead': 'none',
            'constraint': 'true',
            'style': 'invis',
            'weight': '100',
        }

        def add_ordering_edges(node_ids):
            for i in range(len(node_ids) - 1):
                G.edge(node_ids[i], node_ids[i + 1], **invisible_edge_attrs)

        input_node_id = input_node_ids[0] if input_node_ids else None
        field_memory_node_id = field_memory_node_ids[0] if field_memory_node_ids else None

        add_ordering_edges(input_node_ids)
        add_ordering_edges(retrieved_node_ids)
        add_same_rank([node_id
                       for node_id in (concatenate_queries_node_id, concatenated_memory_node_id)
                       if node_id is not None])

        if input_node_id is not None and field_memory_node_id is not None:
            G.edge(input_node_id, field_memory_node_id, **invisible_edge_attrs)
        if combined_scores_node_id is not None:
            for field_memory_node_id in field_memory_node_ids:
                G.edge(field_memory_node_id, combined_scores_node_id, **invisible_edge_attrs)
            for retrieved_node_id in retrieved_node_ids:
                G.edge(combined_scores_node_id, retrieved_node_id, **invisible_edge_attrs)
        for node_id in (concatenate_queries_node_id, concatenated_memory_node_id):
            if node_id is not None:
                for retrieved_node_id in retrieved_node_ids:
                    G.edge(node_id, retrieved_node_id, **invisible_edge_attrs)

        for field in composition.fields:
            if field.weight_node is None:
                continue
            field_memory_node_id = get_node_id(field.memory_node)
            field_weight_node_id = get_node_id(field.weight_node)
            if field_memory_node_id is None or field_weight_node_id is None:
                continue
            add_same_rank([field_memory_node_id, field_weight_node_id])
            G.edge(field_memory_node_id, field_weight_node_id, **invisible_edge_attrs)
