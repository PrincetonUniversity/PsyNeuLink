from __future__ import annotations


def primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def safe_ident(name: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def float_literal(value: float) -> str:
    return repr(float(value))


def zero_vector() -> str:
    return "tl.zeros((BLOCK,), dtype=tl.float32)"
