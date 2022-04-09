from re import S
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

    

class Traveser: 
    def __init__(self, language: str, code: str, language_path = 'build/my-languages.so'):
        self.language = Language(language_path, language)
        self.parser = Parser(self.language)
        self.parser.set_language(self.language)
        self.tree: Tree = self.parser.parse(bytes(code, 'utf8'))
        self.order = 0
        self.nodes = []
        self.edges = []
        self.control_flow_nodes = []
        self.id_map = {}


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
        if node.child_count == 0:
            u = '(%s "%s" %s)' % (IS_LEAF, node.text.decode('utf8'), self.order)
            self.nodes.append(u)
            self.control_flow_nodes.append(u)
            self.id_map[u] = self.order # For later connecting CFG edge
        else:
            u = '(%s %s %s)' % (NOT_LEFT, node.type, self.order)
            self.nodes.append(u)

        # Insert the ID for later queries
        self.setId(node, self.order)

        # AST edge
        if self.order != 0:
            parent_id = self.getId(node.parent)
            self.pushEdge(parent_id, self.order, EdgeType.AST_Forward.name)
            self.pushEdge(self.order, parent_id, EdgeType.AST_Revert)  
        
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
            self.pushEdge(self.id_map[u], self.id_map[v], EdgeType.CFG_Forward.name)
            self.pushEdge(self.id_map[v], self.id_map[u], EdgeType.CFG_Revert.name)



code = open("test.java", "r").read()

 
def main():
    visitor = Traveser('java', code)
    visitor.travesal()
    print(visitor.nodes)
    print(visitor.control_flow_nodes)
    print(visitor.edges)

main()


