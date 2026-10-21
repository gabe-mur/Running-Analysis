"""Bounded, in-memory Garmin ZIP handling. Never extract archive paths to disk."""
from io import BytesIO
from pathlib import PurePosixPath
import stat
import zipfile
import zlib

MAX_ARCHIVE_MEMBERS = 100
MAX_EXPANDED_BYTES = 100 * 1024 * 1024
MAX_MEMBER_BYTES = 50 * 1024 * 1024


def fit_members(content: bytes) -> list[tuple[str, bytes]]:
    """Return only plain FIT members; unrelated archive metadata is ignored.

    Nested archives and compressed FIT members are not recursively unpacked.
    Validate the entire directory before decompressing any FIT payload.
    """
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError('ZIP contains too many entries (maximum 100)')
            if sum(x.file_size for x in members) > MAX_EXPANDED_BYTES:
                raise ValueError('ZIP expanded contents exceed 100 MB')
            fits = []
            for member in members:
                name = member.filename.replace('\\', '/')
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts or ':' in name:
                    raise ValueError('Unsafe path in ZIP')
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError('ZIP symbolic links are not supported')
                if member.flag_bits & 1:
                    raise ValueError('Encrypted ZIP entries are not supported')
                if member.is_dir() or '__MACOSX' in path.parts or path.name.startswith('._'):
                    continue
                if name.casefold().endswith('.fit'):
                    if not 0 < member.file_size <= MAX_MEMBER_BYTES:
                        raise ValueError('ZIP FIT entry must be nonempty and at most 50 MB')
                    fits.append(member)
            if not fits:
                raise ValueError('ZIP must contain at least one uncompressed .fit file')
            result = []
            for member in sorted(fits, key=lambda x: x.filename):
                with archive.open(member) as stream:
                    data = stream.read(MAX_MEMBER_BYTES + 1)
                if len(data) != member.file_size or len(data) > MAX_MEMBER_BYTES:
                    raise ValueError('ZIP FIT entry size is invalid')
                result.append((member.filename.replace('\\', '/'), data))
            return result
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as exc:
        raise ValueError(f'Cannot read FIT archive: {exc}') from exc
