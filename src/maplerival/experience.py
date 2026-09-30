"""MapleStory EXP helpers for level-aware chart coordinates.

The table is the KMS EXP-to-next-level table (levels 200-299).  Keeping the
calculation on the server gives every client the same scale and avoids making
an extra historical API request for every plotted point.
"""

EXP_TO_NEXT = {
    200: 2207026470, 201: 2471869646, 202: 2768494003, 203: 3100713283,
    204: 3472798876, 205: 3889534741, 206: 4356278909, 207: 4879032378,
    208: 5464516263, 209: 6120258214, 210: 7956335678, 211: 8831532602,
    212: 9803001188, 213: 10881331318, 214: 12078277762, 215: 15701761090,
    216: 17114919588, 217: 18655262350, 218: 20334235961, 219: 22164317197,
    220: 28813612356, 221: 30830565220, 222: 32988704785, 223: 35297914119,
    224: 37768768107, 225: 49099398539, 226: 52536356436, 227: 56213901386,
    228: 60148874483, 229: 64359295696, 230: 83667084404, 231: 86177096936,
    232: 88762409844, 233: 91425282139, 234: 94168040603, 235: 122418452783,
    236: 126091006366, 237: 129873736556, 238: 133769948652, 239: 137783047111,
    240: 179117961244, 241: 184491500081, 242: 190026245083, 243: 195727032435,
    244: 201598843408, 245: 262078496430, 246: 269940851322, 247: 278039076861,
    248: 286380249166, 249: 294971656640, 250: 442457484960, 251: 455731209508,
    252: 469403145793, 253: 483485240166, 254: 497989797370, 255: 512929491291,
    256: 528317376029, 257: 544166897309, 258: 560491904228, 259: 577306661354,
    260: 1731919984062, 261: 1749239183902, 262: 1766731575741, 263: 1784398891498,
    264: 1802242880412, 265: 2342915744535, 266: 2366344901980, 267: 2390008350999,
    268: 2413908434508, 269: 2438047518853, 270: 5412465491853, 271: 5466590146771,
    272: 5521256048238, 273: 5576468608720, 274: 5632233294807, 275: 11377111255510,
    276: 12514822381061, 277: 13766304619167, 278: 15142935081083, 279: 16657228589191,
    280: 33647601750165, 281: 37012361925181, 282: 40713598117699, 283: 44784957929468,
    284: 49263453722414, 285: 99512176519276, 286: 109463394171203, 287: 120409733588323,
    288: 132450706947155, 289: 145695777641870, 290: 294305470836577,
    291: 323736017920234, 292: 356109619712257, 293: 391720581683482,
    294: 430892639851830, 295: 870403132500696, 296: 957443445750765,
    297: 1053187790325841, 298: 1158506569358425, 299: 1737759854037637,
}


def required_exp(level: int) -> int:
    try:
        return EXP_TO_NEXT[level]
    except KeyError as exc:
        raise ValueError(f"레벨 {level}의 필요 경험치 정보가 없습니다.") from exc


def progress_rate(level: int, exp: int) -> float:
    return min(100.0, max(0.0, exp / required_exp(level) * 100))


def build_experience_scale(characters: list[dict]) -> dict | None:
    levels = [point["level"] for character in characters for point in character["history"]]
    levels.extend(character["level"] for character in characters if character.get("level") is not None)
    if not levels:
        return None

    minimum, maximum = min(levels), max(levels)
    requirements = {level: required_exp(level) for level in range(minimum, maximum + 1)}
    starts: dict[int, int] = {minimum: 0}
    for level in range(minimum + 1, maximum + 1):
        starts[level] = starts[level - 1] + requirements[level - 1]

    for character in characters:
        for point in character["history"]:
            point["progressRate"] = progress_rate(point["level"], int(point["exp"]))
            point["chartExp"] = str(starts[point["level"]] + int(point["exp"]))
        level = int(character["level"])
        current_exp = int(character["latestExp"])
        character["currentProgressRate"] = progress_rate(level, current_exp)
        character["currentChartExp"] = str(starts[level] + current_exp)

    ticks = [
        {"level": level, "value": str(starts[level]), "label": f"Lv.{level}"}
        for level in range(minimum, maximum + 1)
    ]
    maximum_value = starts[maximum] + requirements[maximum]
    ticks.append({"level": maximum + 1, "value": str(maximum_value), "label": f"Lv.{maximum + 1}"})
    return {
        "minLevel": minimum,
        "maxLevel": maximum,
        "maxValue": str(maximum_value),
        "ticks": ticks,
    }
