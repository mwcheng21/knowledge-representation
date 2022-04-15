from io import TextIOWrapper
import os
from typing import List, Tuple
from tree_sitter import Language, Parser, Tree
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
    'variable_declarator',
    'local_variable_declaration',
    'constant_declaration',
    'declaration',
    # 'field_access'
]

READ_CATEGORY = [
    'formal_parameter',
    'binary_expression',
    'instanceof_expression',
    'spread_parameter',
    'method_invocation'
]

    

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
        self.sep = ' <S> '



    def setId(self, node, id):
        query = '%s%s%s' % (node.start_point, node.end_point, node.type)
        self.id_map[query] = id


    def getId(self, node):
        query = '%s%s%s' % (node.start_point, node.end_point, node.type)
        return self.id_map[query]


    def pushEdge(self, src: int, dst: int, EdgeType):
        self.edges.append('[%s %s %s]' % (src, dst, EdgeType))

    
    """
    Handling each node
    """
    def visit(self, node):
        if node.type == 'assignment_expression':
            print(node.children, node.text)

        if node.child_count == 0:
            u = '%s %s %s' % (IS_LEAF, node.text.decode('utf8'), self.order)
            self.nodes.append(u)
            self.control_flow_nodes.append(u)
            self.id_map[u] = self.order # For later connecting CFG edge
        else:
            if node.type in self.syntax_map: 
                nodeType = self.syntax_map[node.type]
            else:
                nodeType = self.syntax_node_id
                self.syntax_map[node.type] = nodeType
                self.syntax_node_id += 1

            u = '%s %s %s' % (NOT_LEFT, nodeType, self.order)
            self.nodes.append(u)
           
        # Insert the ID for later queries
        self.setId(node, self.order)
       
        # AST edge
        if self.order != 0:
            parent_id = self.getId(node.parent)
            self.pushEdge(parent_id, self.order, EdgeType.AST_Forward.value)
            self.pushEdge(self.order, parent_id, EdgeType.AST_Revert.value)  
        
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
        for leaf in range(1, len(self.control_flow_nodes)):
            prev = leaf - 1
            u = self.control_flow_nodes[prev]
            v = self.control_flow_nodes[leaf]
            self.pushEdge(self.id_map[u], self.id_map[v], EdgeType.CFG_Forward.value)
            self.pushEdge(self.id_map[v], self.id_map[u], EdgeType.CFG_Revert.value)
            

    def leaveOnly(self): 
        pass
    

    def fullGraph(self) -> Tuple[str, str]:
        nodes = self.sep.join(self.nodes)
        edges = self.sep.join(self.edges)
        return nodes, edges


    def astEdgeOnly(self):
        pass


    def cfgEdgeOnly(self):
        pass


    def dfgEdgeOnly(self):
        pass

    
    def encode(self, type: str):
        NODE_START = '<V>'
        EDGE_START = '<E>'
    
        if type == 'fullGraph':
            nodes, edges = self.fullGraph()            
        else:
            nodes, edges = self.fullGraph()  

        return '%s %s %s %s\n' % (NODE_START, nodes, EDGE_START, edges)





def encode(sample: str, type = 'fullGraph') -> str:
    visitor = Traveser('java', sample)
    visitor.travesal()
    return visitor.encode(type)
   


def readFile(filePath: str, outFilePath: str) -> Tuple[List[str], TextIOWrapper]:
    samples = open(filePath, 'r').readlines()
    outFile = open(outFilePath, 'w+')
    return samples, outFile


def preprocess(filePath: str, outFilePath: str, type: str):
    assert os.path.exists(filePath)
    assert os.path.isfile(filePath)

    samples, outFile = readFile(filePath, outFilePath)

    for sample in samples:
        encoding = encode(sample, type)
        outFile.write(encoding)


def __main__():
    input_file = os.path.join(os.path.dirname(os.getcwd()), 'data/medium/train/data.buggy_only')
    os.makedirs( os.path.join(os.path.dirname(os.getcwd()), 'encoded-data/medium/train'), exist_ok=True)
    out_file = os.path.join(os.path.dirname(os.getcwd()), 'encoded-data/medium/train/data.buggy_only')
    preprocess(input_file, out_file, 'fullGraph')




