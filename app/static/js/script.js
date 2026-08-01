document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('predict-form');
  if (!form) return;

  const fields = form.querySelectorAll('select, input');

  fields.forEach(function (field) {
    field.addEventListener('input', updateReadout);
    field.addEventListener('change', updateReadout);
  });

  function updateReadout() {
    fields.forEach(function (field) {
      const name = field.getAttribute('name');
      const target = document.querySelector('[data-field="' + name + '"]');
      if (!target) return;

      const value = field.value;
      if (value === '' || value === null) {
        target.textContent = '—';
        target.classList.add('empty');
      } else {
        target.textContent = value;
        target.classList.remove('empty');
      }
    });
  }
});
