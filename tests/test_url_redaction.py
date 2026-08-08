"""The daemon must never write a presigned S3 URL to its log (#112).

Real incident: a segment failed on a 404 and the traceback put AWSAccessKeyId, Signature and
x-amz-security-token into daemon.log — which is bind-mounted to the host and gets pasted into
tickets. The copy in wanly-api#156 had to be handled carefully before it could be quoted.
"""

import logging

import httpx
import pytest

from daemon.queue_client import _raise_with_details, _redact_url

# Structurally identical to a real presigned URL, with entirely invented values.
#
# The first version of this file pasted the shape from an actual failure and left a real
# STS access key id in it, which GitHub secret scanning flagged — a test about not writing
# credentials down, writing a credential down. The placeholders below are deliberately not
# valid-looking: no ASIA/AKIA prefix, nothing that resembles a real signature.
PRESIGNED = (
    "https://wanly-images.s3.amazonaws.com/2026-07-09/00054-1151936895-swapped.png"
    "?AWSAccessKeyId=EXAMPLEKEYIDNOTREAL0&Signature=EXAMPLESIGNATUREVALUE%3D"
    "&x-amz-security-token=EXAMPLESECURITYTOKENVALUE&Expires=1786145348"
)
SECRETS = (
    "AWSAccessKeyId",
    "Signature=",
    "x-amz-security-token",
    "EXAMPLESECURITYTOKENVALUE",
)


def _response(status=404):
    request = httpx.Request("GET", PRESIGNED)
    return httpx.Response(status, text="<Error><Code>NoSuchKey</Code></Error>", request=request)


class TestRedaction:
    def test_keeps_the_useful_part(self):
        out = _redact_url(httpx.URL(PRESIGNED))
        assert out == "https://wanly-images.s3.amazonaws.com/2026-07-09/00054-1151936895-swapped.png"

    def test_drops_every_credential(self):
        out = _redact_url(httpx.URL(PRESIGNED))
        for secret in SECRETS:
            assert secret not in out

    def test_nothing_secret_reaches_the_log(self, caplog):
        with caplog.at_level(logging.ERROR):
            with pytest.raises(httpx.HTTPStatusError):
                _raise_with_details(_response(), "download_file s3://wanly-images/x.png")
        logged = "\n".join(r.getMessage() for r in caplog.records)
        for secret in SECRETS:
            assert secret not in logged, f"{secret} leaked into the log"
        # and it still says what failed
        assert "404" in logged and "wanly-images" in logged

    def test_nothing_secret_reaches_the_exception_message(self):
        """The traceback is the copy that ends up pasted into tickets."""
        with pytest.raises(httpx.HTTPStatusError) as exc:
            _raise_with_details(_response(), "download_file s3://wanly-images/x.png")
        message = str(exc.value)
        for secret in SECRETS:
            assert secret not in message, f"{secret} leaked into the exception"
        assert "404" in message

    def test_still_raises_httpstatuserror_so_callers_are_unaffected(self):
        with pytest.raises(httpx.HTTPStatusError):
            _raise_with_details(_response(500), "upload_segment_output abc")
