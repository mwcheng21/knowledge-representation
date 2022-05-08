from io import TextIOWrapper
import os
from typing import List, Tuple
from tree_sitter import Language, Parser
from enum import Enum


Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'vendor/tree-sitter-javascript',
    'vendor/tree-sitter-python',
    'vendor/tree-sitter-java',
    'vendor/tree-sitter-c-sharp',
  ]
)

IS_LEAF = 1
NOT_LEFT = 0

IS_PATCH = 1
NOT_PATCH = 0

PATCH_START = "/*#-PATCH-START-#*/"
PATCH_END = "/*#-PATCH-END-#*/"


class EdgeType(Enum):
    CFG_Forward    = 1
    CFG_Revert     = 2
    AST_Forward    = 3
    AST_Revert     = 4 
    LAST_READ      = 5
    LAST_WRITE     = 6 


WRITE_CATEGORY = [
    'enum_declaration',
    'enum_constant',
    'class_declaration',
    'constructor_declaration',
    'method_declaration',
    'field_declaration',
    'interface_declaration',
    'record_declaration',
    'annotation_type_declaration',
]

ARTBITRARY = [
    'assignment_expression',
]

READ_CATEGORY = [
    'formal_parameter',
    'binary_expression',
    'instanceof_expression',
    'spread_parameter',
    'method_invocation',
    'class_literal',
    'object_creation_expression'
]

 # strip the !!
def strip_edge(edge): 
    edge_rest = edge[0: -3]
    edgeType = edge[-3:][1]
    return edge_rest + edgeType

def strip_node(node): 
    node_rest = node[3:]
    nodeType = node[0:3][1]
    return nodeType + node_rest
    

class Traveser: 
    def __init__(self, language: str, code: str, language_path = 'build/my-languages.so'):
        self.language = Language(language_path, language)
        self.parser = Parser(self.language)
        self.parser.set_language(self.language)
        self.tree = self.parser.parse(bytes(code, 'utf8'))
        self.order = 0
        self.nodes = []
        self.edges = []
        self.control_flow_nodes = []
        self.id_map = {}
        self.syntax_node_id = 0
        self.syntax_map = {}
        self.sep = ' <s> '
        self.read_nodes = {}
        self.write_nodes = {}
        self.patch_tokens = []
        self.isPatch = False
        

    def setId(self, node, id):
        query = '%s%s%s' % (node.start_point, node.end_point, node.type)
        self.id_map[query] = id
        return query


    def getId(self, node) -> int:
        query = '%s%s%s' % (node.start_point, node.end_point, node.type)
        if query in self.id_map:
            return self.id_map[query]
        else:
            return -1


    def pushEdge(self, src: int, dst: int, EdgeType):
        self.edges.append('%s %s !%s!' % (src, dst, EdgeType))

    
    def dataFlowEdge(self, node, type: str): 
        assert node.type == 'identifier' or node.type == 'type_identifier'

        variable = node.text.decode('utf8')
        
        node_id = self.getId(node)

        if type == EdgeType.LAST_READ.name:
            if variable in self.read_nodes: 
                for dst in self.read_nodes[variable]:
                    self.pushEdge(node_id, dst, EdgeType.LAST_READ.value)
                self.read_nodes[variable].append(node_id)

            else:
                self.read_nodes[variable] = []
        
        if type == EdgeType.LAST_WRITE.name:
            if variable in self.write_nodes:
                for dst in self.write_nodes[variable]:
                    self.pushEdge(node_id, dst, EdgeType.LAST_WRITE.value)
                self.pushEdge(node_id, node_id, EdgeType.LAST_WRITE.value) # Self connected
                self.write_nodes[variable].append(node_id)
            else:
                self.write_nodes[variable] = [node_id]
                self.read_nodes[variable] = [node_id]
        pass



    def buildDataFlow(self, node): 
        if node.parent.type in WRITE_CATEGORY:
            self.dataFlowEdge(node, EdgeType.LAST_WRITE.name)
        elif node.parent.type in READ_CATEGORY:
            self.dataFlowEdge(node, EdgeType.LAST_READ.name)
        elif node.parent.type in ARTBITRARY:
            '''
            If it is the first children, it is the writtern variable
            '''
            if self.getId(node.parent.children[0]) == self.getId(node): 
                self.dataFlowEdge(node, EdgeType.LAST_WRITE.name)
            else:
                self.dataFlowEdge(node, EdgeType.LAST_READ.name)

        else:
            pass

    
    """
    Handling each node
    """
    def visit(self, node):
        # Insert the ID for later queries
        self.setId(node, self.order)

        # Make sure Turn it off before actuall handling
        if node.type == 'block_comment' and node.text.decode().strip() == PATCH_END.strip(): 
            self.isPatch = False 


        patch_signal = IS_PATCH if self.isPatch else NOT_PATCH


        if node.child_count == 0:
            token_val = node.text.decode('utf8')
            u = '!%s! %s %s %s' % (IS_LEAF, token_val, self.order, patch_signal)
            self.nodes.append(u)
            self.control_flow_nodes.append(u)
            self.id_map[u] = self.order # For later connecting CFG edge
            if self.isPatch: self.patch_tokens.append(f'{token_val} {self.order}')
            if node.type == 'identifier' or node.type == 'type_identifier':
                self.buildDataFlow(node)
    
        else:
            if node.type in self.syntax_map: 
                nodeType = self.syntax_map[node.type]
            else:
                nodeType = self.syntax_node_id
                self.syntax_map[node.type] = nodeType
                self.syntax_node_id += 1

            u = '!%s! %s %s %s' % (NOT_LEFT, nodeType, self.order, patch_signal)
            self.nodes.append(u)
        
       
        # AST edge
        if self.order != 0:
            parent_id = self.getId(node.parent)
            self.pushEdge(parent_id, self.order, EdgeType.AST_Forward.value)
            # self.pushEdge(self.order, parent_id, EdgeType.AST_Revert.value)  


        if node.type == 'block_comment' and node.text.decode().strip() == PATCH_START.strip(): 
            self.isPatch = True 

        self.order += 1


    """
    Pre-order DFS
    """
    def travesal(self):
        hit_root = False
        cursor = self.tree.walk()
        while not hit_root:
            self.visit(cursor.node)

            if cursor.goto_first_child():
                continue

            if cursor.goto_next_sibling():
                continue
            
            find_next_rhs = True

            while find_next_rhs:
                if not cursor.goto_parent():
                    find_next_rhs = False
                    hit_root = True

                if cursor.goto_next_sibling():
                    find_next_rhs = False


        # Handle the CFG edge in the end
        # for leaf in range(1, len(self.control_flow_nodes)):
        #     prev = leaf - 1
        #     u = self.control_flow_nodes[prev]
        #     v = self.control_flow_nodes[leaf]
        #     self.pushEdge(self.id_map[u], self.id_map[v], EdgeType.CFG_Forward.value)
        #     # self.pushEdge(self.id_map[v], self.id_map[u], EdgeType.CFG_Revert.value)
        

    def leaveOnly(self): 
        leaves = filter(lambda node: ('!%s!' % IS_LEAF) in node, self.nodes)
        nodes = list(leaves)

        leavesEdges = filter(lambda edge: ('!%s!' % EdgeType.AST_Forward.value) not in edge \
                                        and ('!%s!' % EdgeType.AST_Revert.value) not in edge, self.edges)
        edges = list(leavesEdges)

        return nodes, edges
        
    

    def fullGraph(self) -> Tuple[str, str]:
        return self.nodes, self.edges


    def astEdgeOnly(self):
        nodes = self.nodes

        astEdge = filter(lambda edge: ('!%s!' % EdgeType.AST_Forward.value) in edge \
                                        or ('!%s!' % EdgeType.AST_Revert.value) in edge, self.edges)

        edges = list(astEdge)

        return nodes, edges
        


    def cfgEdgeOnly(self):
        nodes, edges = self.leaveOnly()

        cfgEdges = filter(lambda edge: ('!%s!' % EdgeType.LAST_READ.value) not in edge \
                                        and ('!%s!' % EdgeType.LAST_WRITE.value) not in edge, edges)
        
        edges = list(cfgEdges)
        return nodes, edges


    def dfgEdgeOnly(self):
        nodes, edges = self.leaveOnly()

        dfgEdges = filter(lambda edge: ('!%s!' % EdgeType.CFG_Forward.value) not in edge \
                                        and ('!%s!' % EdgeType.CFG_Revert.value) not in edge, edges)
        
        edges = list(dfgEdges)
        return nodes, edges

    
    def encode(self, type: str = 'fullGraph'):
        NODE_START = '<V>'
        EDGE_START = '<E>'
    
        if type == 'fullGraph':
            nodes, edges = self.fullGraph()            
        elif type == 'leaveOnly':
            nodes, edges = self.leaveOnly()        
        elif type == 'astOnly':
            nodes, edges = self.astEdgeOnly()      
        elif type == 'cfgOnly':
            nodes, edges = self.cfgEdgeOnly()    
        elif type == 'dfgOnly':
            nodes, edges = self.dfgEdgeOnly()       
        else:
            raise KeyError("No found type")


        nodes = self.sep.join(list(map(strip_node, nodes)))
        edges = self.sep.join(list(map(strip_edge, edges)))

        return '%s %s %s %s\n' % (NODE_START, nodes, EDGE_START, edges)


def encode(sample: str, type = 'fullGraph') -> str:
    visitor = Traveser('java', sample)
    visitor.travesal()
    return visitor.encode(type), ' '.join(visitor.patch_tokens)
   

def readFile(filePath: str, outFilePath: str) -> Tuple[List[str], TextIOWrapper]:
    samples = open(filePath, 'r').readlines()
    outFile = open(outFilePath, 'w+')
    return samples, outFile


def preprocess(filePath: str, outFilePath: str, out_buggy: str, type: str):
    assert os.path.exists(filePath)
    assert os.path.isfile(filePath)

    samples, outFile = readFile(filePath, outFilePath)
    out_buggy_fd = open(out_buggy, 'w+')

    for sample in samples:
        encoding, patch_lw = encode(sample, type)
        outFile.write(encoding)
        out_buggy_fd.write(patch_lw + '\n')




