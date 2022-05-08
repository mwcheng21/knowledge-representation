import os


PATCH_START = " /*#-PATCH-START-#*/ "
PATCH_END = " /*#-PATCH-END-#*/ "



def highlight(contextPath: str, patchPath: str, outFilePath: str):
    assert os.path.exists(contextPath)
    assert os.path.exists(patchPath)
    

    context_fp = open(contextPath, 'rt')
    patch_fp = open(patchPath, 'rt')
    out_fp = open(outFilePath, 'w+')

    contexts = context_fp.readlines()
    patches = patch_fp.readlines()
    lines = len(contexts)

    assert lines == len(patches)

    for i in range(lines): 
        ctx: str = contexts[i].strip()
        patch: str = patches[i].strip()
        patch_start = ctx.find(patch)
        patch_end = patch_start + len(patch)

        if patch == '<EMPTY>' or ctx[patch_start:patch_end] != patch:
            out_fp.write(ctx + '\n')
        else:
            hl_ctx = ctx[:patch_start] + PATCH_START + ctx[patch_start:patch_end] + PATCH_END + ctx[patch_end:] + '\n'
            out_fp.write(hl_ctx)

        
    
def process(set = 'train'):
    ctx_file = os.path.join('./', 'original-data/small/%s/data.parent_code' % set)
    patch_file = os.path.join('./' 'original-data/small/%s/data.parent_buggy_only' % set)
    output_file = os.path.join('./', 'original-data/small/%s/data.parent_full_code_hl' % set)
    
    highlight(ctx_file, patch_file, output_file)


process('test')