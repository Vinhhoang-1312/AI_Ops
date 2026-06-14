"""Map visible expression labels to call-center coaching cues."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentCue:
    label: str
    display_name: str
    headline: str
    tone: str
    action: str
    suggested_response: str


_CUES = {
    "angry": AgentCue(
        label="angry",
        display_name="Tuc gian / kho chiu",
        headline="Khach co dau hieu buc bo. Uu tien ha nhiet cuoc goi.",
        tone="Cham, binh tinh, khong tranh luan.",
        action="Xac nhan van de, xin loi neu trai nghiem chua tot, dua ra buoc tiep theo ro rang.",
        suggested_response="Em rat tiec vi anh/chi phai mat thoi gian voi viec nay. De em kiem tra ngay va dua ra cach xu ly cu the.",
    ),
    "disgust": AgentCue(
        label="disgust",
        display_name="Khong hai long",
        headline="Khach co ve that vong. Can thua nhan cam giac cua ho.",
        tone="Chan thanh, ngan gon, tap trung sua loi.",
        action="Hoi dung diem gay kho chiu va de xuat mot hanh dong khac phuc.",
        suggested_response="Em hieu dieu nay co the rat kho chiu. Anh/chi co the cho em biet phan nao lam minh that vong nhat khong?",
    ),
    "contempt": AgentCue(
        label="contempt",
        display_name="Khinh thuong / kho chiu",
        headline="Khach co dau hieu thieu tin tuong. Can giu binh tinh va lay lai su ro rang.",
        tone="Binh tinh, ton trong, khong phong thu.",
        action="Xac nhan lai dieu khach chua hai long, dua bang chung hoac lua chon xu ly cu the.",
        suggested_response="Em hieu anh/chi co the chua yen tam. De em noi ro lai phan nay va dua ra lua chon xu ly cu the.",
    ),
    "fear": AgentCue(
        label="fear",
        display_name="Lo lang",
        headline="Khach co dau hieu lo lang. Hay lam ro va tao cam giac an toan.",
        tone="Am ap, chac chan, tranh dung tu gay ap luc.",
        action="Noi ro quy trinh, chia nho van de, xac nhan ban se dong hanh den khi xong.",
        suggested_response="Minh di tung buoc nhe. Em se o day ho tro den khi anh/chi nam ro cach xu ly.",
    ),
    "happy": AgentCue(
        label="happy",
        display_name="Vui ve",
        headline="Khach dang phan hoi tich cuc. Co the giu nhip hoi thoai nhe nhang.",
        tone="Than thien, tu tin, van chuyen nghiep.",
        action="Duy tri nang luong tot, xac nhan nhu cau tiep theo va ket thuc gon neu da xong.",
        suggested_response="Rat vui vi minh da xu ly duoc viec nay. Anh/chi con muon em kiem tra them phan nao khong?",
    ),
    "sad": AgentCue(
        label="sad",
        display_name="Buon / met moi",
        headline="Nguoi dung co dau hieu buon. Hay dong cam va khuyen khich chia se.",
        tone="Nhe, cham, nhieu su lang nghe.",
        action="Dong cam, cheer up nhe nhang, hoi mot cau mo, tranh day nhanh sang giai phap neu ho chua san sang.",
        suggested_response="Em rat tiec vi anh/chi dang gap dieu nay. Minh co the chia se them de em ho tro dung hon khong?",
    ),
    "sleepy": AgentCue(
        label="sleepy",
        display_name="Met moi / thieu tap trung",
        headline="Khach co ve met moi. Nen noi ngan gon va giam tai thong tin.",
        tone="Cham, ngan gon, ro y.",
        action="Tom tat thanh tung buoc nho, hoi xac nhan truoc khi chuyen sang buoc tiep theo.",
        suggested_response="Em se noi ngan gon tung buoc nhe. Truoc tien minh xac nhan lai phan quan trong nhat la...",
    ),
    "surprise": AgentCue(
        label="surprise",
        display_name="Bat ngo",
        headline="Khach co ve bat ngo. Can lam ro thong tin vua noi.",
        tone="Ro rang, cham lai mot nhip.",
        action="Tom tat lai diem chinh va hoi xem phan nao can giai thich them.",
        suggested_response="Em se noi lai ngan gon de minh de theo doi: diem quan trong la...",
    ),
    "neutral": AgentCue(
        label="neutral",
        display_name="Trung tinh",
        headline="Cam xuc hien tai kha trung tinh. Tiep tuc ho tro theo quy trinh.",
        tone="Chuyen nghiep, ro rang, than thien vua du.",
        action="Hoi muc tieu tiep theo va dua ra lua chon ngan gon.",
        suggested_response="Em da nam duoc thong tin chinh. Buoc tiep theo minh muon xu ly phan nao truoc?",
    ),
}

_UNKNOWN = AgentCue(
    label="unknown",
    display_name="Chua ro",
    headline="Chua du tin hieu on dinh de ket luan.",
    tone="Binh tinh, quan sat them.",
    action="Tiep tuc lang nghe va doi them mau frame.",
    suggested_response="Em dang lang nghe. Anh/chi co the chia se them mot chut ve van de minh dang gap khong?",
)


def cue_for_expression(label: str | None, confidence: float | None = None) -> AgentCue:
    """Return a coaching cue for a visible expression label."""
    if not label:
        return _UNKNOWN

    normalized = label.strip().lower()
    cue = _CUES.get(normalized, _UNKNOWN)

    if confidence is not None and confidence < 0.45 and cue is not _UNKNOWN:
        return AgentCue(
            label=cue.label,
            display_name=f"{cue.display_name}?",
            headline="Tin hieu con yeu. Hay xem day la goi y, khong phai ket luan.",
            tone=cue.tone,
            action=cue.action,
            suggested_response=cue.suggested_response,
        )
    return cue
