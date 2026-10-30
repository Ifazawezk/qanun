from qanun.cli import build_parser


def test_ask_leaves_model_to_the_environment():
    args = build_parser().parse_args(["ask", "annual", "leave"])
    assert args.model is None
    assert args.question == ["annual", "leave"]
    assert args.mode == "auto"


def test_index_and_eval_accept_several_models():
    args = build_parser().parse_args(["eval", "--model", "e5-small", "--model", "minilm"])
    assert args.model == ["e5-small", "minilm"]
    assert build_parser().parse_args(["index"]).model is None


def test_serve_defaults():
    args = build_parser().parse_args(["serve"])
    assert (args.host, args.port) == ("127.0.0.1", 8000)
