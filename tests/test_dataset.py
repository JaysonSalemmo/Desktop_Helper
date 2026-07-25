from model.data.tool_calls import generate
from model.dataset import result_span_mask
from model.tokenizer import DesktopHelperTokenizer

RS, RE = 98, 99  # stand-in result_start / result_end ids


def test_result_span_mask_covers_block_inclusive():
    #      0    1    2      3     4   5   6     7    8    9
    ids = [1,   50,  51,    5,    RS, 60, 61,   RE,  70,  2]
    #      bos  prompt....   CALL  [R  ...result  R]  reply eos
    mask = result_span_mask(ids, RS, RE)
    # targets y[k] = ids[k+1]; masked targets are RS, 60, 61, RE → indices 3-6
    assert mask == [False, False, False, True, True, True, True, False, False]


def test_result_span_mask_keeps_call_and_reply():
    ids = [1, 50, 5, RS, 60, RE, 70, 2]
    mask = result_span_mask(ids, RS, RE)
    assert mask[1] is False  # target ids[2] = the CALL token — routing signal stays
    assert mask[5] is False  # target ids[6] = first reply token — stays
    assert mask[2] and mask[3] and mask[4]  # RS, 60, RE — masked


def test_result_span_mask_no_result_block():
    ids = [1, 50, 51, 70, 2]
    assert result_span_mask(ids, RS, RE) == [False] * 4


def test_generator_examples_parse_and_roundtrip():
    tok = DesktopHelperTokenizer.load()
    examples = generate(300, seed=7)
    tools_seen = set()
    chat_seen = 0
    for ex in examples:
        if not ex["response"].startswith("[CALL: "):
            # no-tool chat example: plain reply, no protocol tokens at all
            chat_seen += 1
            ids = tok.encode(ex["response"])
            assert tok.result_start_id not in ids
            assert all(tok.is_tool_call(i) is None for i in ids)
            continue
        tool = ex["response"].split("]")[0].removeprefix("[CALL: ")
        tools_seen.add(tool)
        # every tool token must resolve; every result block must be delimited
        ids = tok.encode(ex["response"])
        assert tok.tool_token_id(tool) in ids
        assert ids.count(tok.result_start_id) == 1
        assert ids.count(tok.result_end_id) == 1
    assert len(tools_seen) == 10
    assert chat_seen > 0  # the routing-contrast category must be present


def test_generator_content_is_high_entropy():
    # the old failure mode: small fixed pools → repeated results. sample results
    # across two seeds and require near-uniqueness
    results = []
    for ex in generate(400, seed=11):
        if "[RESULT]" not in ex["response"]:
            continue  # chat examples have no result block
        result = ex["response"].split("[RESULT]")[1].split("[/RESULT]")[0]
        results.append(result)
    unique = len(set(results)) / len(results)
    assert unique > 0.85, f"result content repeats too much ({unique:.0%} unique)"
