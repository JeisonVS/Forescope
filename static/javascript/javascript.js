var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl)
})

function animateresultado(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        obj.innerHTML = Math.floor(progress * (end - start) + start);
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

const obj = document.getElementById("resultado");
animateresultado(obj, 0, pronostico_dia, 1000);

document.getElementById('downloadButton').addEventListener('click', () => {
    const table = document.getElementById('reportTable');
    const rows = table.querySelectorAll('tr');
    let csvContent = "Tabla de Reportes del Mes\n";

    rows.forEach(row => {
        const rowData = [];
        row.querySelectorAll('th, td').forEach(cell => {
            rowData.push(cell.innerText);
        });
        csvContent += rowData.join(';') + '\n';
    });

    const blob = new Blob([csvContent], {type: 'text/csv'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'reportes.csv';
    link.click();
});

// img perfil navbar
function handleImageError() {
    var url_img_default = "/static/img/perfil_default.jpg";
    var img_perfil = document.querySelectorAll(".img-perfil");
    img_perfil.forEach(function (imagen) {
        imagen.src = url_img_default;
    });
}

//  fin img perfil navbar

// subirImagen desde tag img
function subirImagen() {
    // Manejar el cambio en el campo de carga de archivo
    document.getElementById('fileInput').addEventListener('change', function () {
        const fileInput = this;
        const imagePreview = document.getElementById('imagePreview');

        // Verificar si se seleccionó un archivo
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];

            // Verificar si el archivo es una imagen
            if (file.type.startsWith('image/')) {
                // Crear una URL de objeto para previsualizar la imagen
                const objectURL = URL.createObjectURL(file);
                imagePreview.src = objectURL;
            }
        }
    });
}
// fin subirImagen desde tag img
