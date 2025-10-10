function searchUsers() {
    const input = document.getElementById('searchInput');
    const filter = input.value.toLowerCase();
    const table = document.getElementById('usersTableBody');
    const rows = table.getElementsByTagName('tr');

    for (let i = 0; i < rows.length; i++) {
        const cells = rows[i].getElementsByTagName('td');
        let found = false;

        for (let j = 0; j < cells.length - 1; j++) {
            if (cells[j].textContent.toLowerCase().indexOf(filter) > -1) {
                found = true;
                break;
            }
        }

        rows[i].style.display = found ? '' : 'none';
    }
}

function changePassword(userId, userName) {
    const modal = document.getElementById('passwordModal');
    const form = document.getElementById('passwordForm');
    
    form.action = `/admin/change-password/${userId}`;
    modal.style.display = 'block';
}

function closeModal() {
    document.getElementById('passwordModal').style.display = 'none';
}

function deleteUser(userId, userName) {
    const modal = document.getElementById('deleteModal');
    const form = document.getElementById('deleteForm');
    const message = document.getElementById('deleteMessage');
    
    message.textContent = `Are you sure you want to delete ${userName}'s account? This action cannot be undone.`;
    form.action = `/admin/delete-user/${userId}`;
    modal.style.display = 'block';
}

function closeDeleteModal() {
    document.getElementById('deleteModal').style.display = 'none';
}

// Close modals when clicking outside
window.onclick = function(event) {
    const passwordModal = document.getElementById('passwordModal');
    const deleteModal = document.getElementById('deleteModal');
    
    if (event.target == passwordModal) {
        passwordModal.style.display = 'none';
    }
    if (event.target == deleteModal) {
        deleteModal.style.display = 'none';
    }
}
