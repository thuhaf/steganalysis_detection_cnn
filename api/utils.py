from pathlib import Path
from fastapi import UploadFile
import tempfile
import shutil
from typing import List
import uuid


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save uploaded file to temporary location

    Args:
        upload_file: FastAPI UploadFile

    Returns:
        Path to saved file
    """
    # Create temp directory if not exists
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)

    # Generate unique filename
    suffix = Path(upload_file.filename).suffix
    temp_path = temp_dir / f"{uuid.uuid4()}{suffix}"

    # Save file
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return str(temp_path)


def cleanup_temp_files(file_paths: List[str]):
    """Delete temporary files"""
    for path in file_paths:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")


def generate_report(batch_result) -> str:
    """
    Generate PDF report from batch results

    Args:
        batch_result: BatchDetectionResult

    Returns:
        Path to generated PDF
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    # Create temp PDF
    pdf_path = Path("temp") / f"report_{uuid.uuid4()}.pdf"

    # Create document
    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph("Steganalysis Detection Report", styles['Title'])
    elements.append(title)

    # Summary
    summary = f"""
    <b>Summary:</b><br/>
    Total Files: {batch_result.total_files}<br/>
    Stego Detected: {batch_result.stego_detected}<br/>
    Cover Images: {batch_result.cover_detected}<br/>
    Total Time: {batch_result.total_time:.2f}s<br/>
    """
    elements.append(Paragraph(summary, styles['Normal']))

    # Table of results
    data = [['File', 'Prediction', 'Confidence']]
    for result in batch_result.results:
        data.append([
            Path(result.file_path).name if result.file_path else 'N/A',
            result.prediction,
            f"{result.confidence:.4f}"
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)

    # Build PDF
    doc.build(elements)

    return str(pdf_path)
